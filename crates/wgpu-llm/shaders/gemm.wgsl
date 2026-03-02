@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage,read_write> c: array<f32>;

@compute @workgroup_size({{WG_X}}, {{WG_Y}})
fn main (
  @builtin(global_invocation_id) gid: vec3<u32>
) {
  let row = gid.x;
  let col = gid.y;

  if (row >= {{M}}u || col >= {{N}}u) {
    return;
  }

  var sum: f32 = 0.0;
  for (var i: u32 = 0u; i < {{K}}u; i = i + 1u) {
    sum = sum + a[row * {{K}}u + i] * b[i * {{N}}u + col];
  }

  c[row * {{N}}u + col] = sum;
}

