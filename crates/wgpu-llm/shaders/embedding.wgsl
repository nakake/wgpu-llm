@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read> token_embedding: array<f32>;
@group(0) @binding(2) var<storage, read> position_embedding: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size({{WG_SIZE}})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let index = gid.x;
    if (index >= {{NUM_TOKENS}}u) {
        return;
    }

    let HIDDEN_SIZE: u32 = {{HIDDEN_SIZE}}u;

    let offset = index * HIDDEN_SIZE;
    let token_id = token_ids[index];
    for (var i: u32 = 0u; i < HIDDEN_SIZE; i = i + 1u) {
        let token_emb = token_embedding[token_id * HIDDEN_SIZE + i];
        let pos_emb = position_embedding[index * HIDDEN_SIZE + i];
        output[offset + i] = token_emb + pos_emb;
    }
}