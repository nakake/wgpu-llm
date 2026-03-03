@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const HIDDEN_SIZE: u32 = {{HIDDEN_SIZE}}u;

@compute @workgroup_size({{WG_SIZE}})
fn main (
  @builtin(global_invocation_id) gid: vec3<u32>
) {
  if (gid.x >= {{NUM_TOKENS}}u) {
    return;
  }

  let offset = gid.x * HIDDEN_SIZE;

  var mean: f32 = 0.0;
  for (var i: u32 = 0u; i < HIDDEN_SIZE; i = i + 1u) {
    mean = mean + input[offset + i];
  }
  mean = mean / f32(HIDDEN_SIZE);

  var variance: f32 = 0.0; 
  for (var i: u32 = 0u; i < HIDDEN_SIZE; i = i + 1u) {
    let diff = input[offset + i] - mean;
    variance = variance + diff * diff;
  }
  variance = variance / f32(HIDDEN_SIZE); 

  for (var i: u32 = 0u; i < HIDDEN_SIZE; i = i + 1u) {
    let norm = (input[offset + i] - mean) / sqrt(variance + 1e-5);
    output[offset + i] = norm * gamma[i] + beta[i];
  }
}
