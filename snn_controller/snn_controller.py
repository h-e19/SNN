from controller import Robot
import random
import csv
import time
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional

# --- SNN Definition ---
class LineSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = layer.Linear(3, 16)
        self.lif1 = neuron.LIFNode(tau=2.0)
        self.fc2 = layer.Linear(16, 2)
        self.lif2 = neuron.LIFNode(tau=2.0)

    def forward(self, x):
        x = self.lif1(self.fc1(x))
        x = self.lif2(self.fc2(x))
        return x

# --- Robot Setup ---
robot = Robot()
timestep = int(robot.getBasicTimeStep())

gs0 = robot.getDevice('gs0')
gs1 = robot.getDevice('gs1')
gs2 = robot.getDevice('gs2')
for gs in [gs0, gs1, gs2]:
    gs.enable(timestep)

left_motor  = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

MAX_SPEED = 6.28

# --- SNN + STDP setup ---
snn = LineSNN()
functional.set_step_mode(snn, step_mode='s')

lr         = 0.05
trace_pre  = torch.zeros(3)
trace_post = torch.zeros(2)
tau_trace  = 0.9

# --- Weight init (once, before the loop) ---
with torch.no_grad():
    snn.fc1.weight.data = torch.randn(16, 3) * 2.0
    snn.fc2.weight.data = torch.randn(2, 16) * 2.0

stuck_counter = 0

# --- Logging setup ---
log_filename = f"snn_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
log_file = open(log_filename, 'w', newline='')
log_writer = csv.writer(log_file)
log_writer.writerow([
    'timestep', 'gs0', 'gs1', 'gs2', 'line_signal', 'error',
    'stuck_counter', 'reward', 'v_out_left', 'v_out_right',
    'left_vel', 'right_vel', 'dead_end_triggered'
])
step_count = 0

# --- Main Loop ---
while robot.step(timestep) != -1:
    step_count += 1

    # 1. read sensors, normalize to [0, 1]
    raw = torch.tensor([gs0.getValue(), gs1.getValue(), gs2.getValue()])
    sensors = (raw / 1000.0).clamp(0, 1)

    # dead-end detection: stuck if all sensors read high (off the line) for too long
    if sensors.min().item() > 0.75:
        stuck_counter += 1
    else:
        stuck_counter = 0

    if stuck_counter > 50:
        print("DEAD END - picking random direction")
        log_writer.writerow([step_count, '', '', '', '', '', stuck_counter, '', '', '', '', '', 1])
        log_file.flush()

        turn_time = random.uniform(0.3, 1.2)   # seconds
        turn_dir  = random.choice([-1, 1])     # -1 = turn left, 1 = turn right

        left_motor.setVelocity(turn_dir * MAX_SPEED)
        right_motor.setVelocity(-turn_dir * MAX_SPEED)

        steps = int((turn_time * 1000) / timestep)
        for _ in range(steps):
            if robot.step(timestep) == -1:
                break

        stuck_counter = 0
        continue

    # 2. convert to spikes
    spikes_in = (sensors < 0.5).float().unsqueeze(0)

    # 3. forward pass
    functional.reset_net(snn)
    spikes_out = snn(spikes_in).squeeze(0)

    # decode membrane potential for continuous motor signal
    v_out = snn.lif2.v.squeeze(0).detach()

    # 4. error-based steering + SNN output
    error      = sensors[2].item() - sensors[0].item()
    correction = error * 0.5 * MAX_SPEED
    baseline   = 0.4 * MAX_SPEED

    left_vel  = baseline + v_out[0].item() * MAX_SPEED - correction
    right_vel = baseline + v_out[1].item() * MAX_SPEED + correction

    left_vel  = max(-MAX_SPEED, min(MAX_SPEED, left_vel))
    right_vel = max(-MAX_SPEED, min(MAX_SPEED, right_vel))

    left_motor.setVelocity(left_vel)
    right_motor.setVelocity(right_vel)

    # 5. reward: is the line detected SOMEWHERE under the sensor array, regardless of shape?
    left_val, center_val, right_val = sensors[0].item(), sensors[1].item(), sensors[2].item()

    darkest    = min(left_val, center_val, right_val)   # most "on line" reading
    brightest  = max(left_val, center_val, right_val)    # most "off line" reading

    # on_line: something is genuinely dark (line detected somewhere)
    on_line = 1.0 - darkest

    # spread: at least one sensor is clearly off the line too (not fully engulfed in black)
    spread = brightest - darkest

    # both conditions matter: dark spot present AND not a uniform black blob
    line_signal = on_line * spread

    # penalty grows the longer it's been off the line, giving STDP a real gradient to avoid this
    lost_penalty = 0.02 * stuck_counter

    reward = line_signal - lost_penalty

    print(f"gs: {sensors.tolist()} | line_signal: {line_signal:.3f} | stuck: {stuck_counter} | reward: {reward:.3f} | v_out: {v_out.tolist()}")

    log_writer.writerow([
        step_count, left_val, center_val, right_val, line_signal, error,
        stuck_counter, reward, v_out[0].item(), v_out[1].item(),
        left_vel, right_vel, 0
    ])
    log_file.flush()

    # 6. reward-modulated STDP
    trace_pre  = tau_trace * trace_pre  + spikes_in.squeeze(0)
    trace_post = tau_trace * trace_post + spikes_out.detach()

    # trace values can compound if traces keep accumulating without ever fully decaying;
    # clamp them so they can't runaway
    trace_pre  = trace_pre.clamp(-10, 10)
    trace_post = trace_post.clamp(-10, 10)

    with torch.no_grad():
        dw1 = reward * lr * torch.outer(snn.lif1.v.squeeze(0).detach(), trace_pre)
        dw2 = reward * lr * torch.outer(trace_post, snn.lif1.v.squeeze(0).detach())

        # guard against nan/inf appearing in the update itself
        dw1 = torch.nan_to_num(dw1, nan=0.0, posinf=0.0, neginf=0.0)
        dw2 = torch.nan_to_num(dw2, nan=0.0, posinf=0.0, neginf=0.0)

        snn.fc1.weight += dw1
        snn.fc2.weight += dw2

        # clamp weights so they can't grow unbounded and blow up membrane potentials
        snn.fc1.weight.clamp_(-5.0, 5.0)
        snn.fc2.weight.clamp_(-5.0, 5.0)

        # if weights somehow already contain nan/inf (e.g. from before this fix), reset them
        if torch.isnan(snn.fc1.weight).any() or torch.isinf(snn.fc1.weight).any():
            print("WARNING: fc1 weights corrupted, reinitializing")
            snn.fc1.weight.data = torch.randn(16, 3) * 2.0
        if torch.isnan(snn.fc2.weight).any() or torch.isinf(snn.fc2.weight).any():
            print("WARNING: fc2 weights corrupted, reinitializing")
            snn.fc2.weight.data = torch.randn(2, 16) * 2.0

    functional.reset_net(snn)