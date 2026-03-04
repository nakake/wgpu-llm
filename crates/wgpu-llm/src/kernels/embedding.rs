use std::collections::HashMap;

use crate::template;

pub struct Embedding {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    hidden_size: u32,
    num_tokens: u32,
    workgroup_size: u32,
}

impl Embedding {
    pub fn new(
        device: &wgpu::Device,
        hidden_size: u32,
        num_tokens: u32,
    ) -> Self {
        let shader = include_str!("../../shaders/embedding.wgsl");
        let mut map = HashMap::new();
        let workgroup_size = 256;
        map.insert("HIDDEN_SIZE", hidden_size.to_string());
        map.insert("NUM_TOKENS", num_tokens.to_string());
        map.insert("WG_SIZE", workgroup_size.to_string());
        let replace_shader = template::render(shader, &map);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("embedding shader"),
            source: wgpu::ShaderSource::Wgsl(replace_shader.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("embedding bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("embedding pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("embedding pipeline layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { 
            pipeline, 
            bind_group_layout, 
            hidden_size, 
            num_tokens, 
            workgroup_size
        }
    }

    pub fn dispatch(
        &self,
        token_ids: &wgpu::Buffer,
        token_embed: &wgpu::Buffer,
        position_embed: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        device: &wgpu::Device,
        pass: &mut wgpu::ComputePass<'_>,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("embedding bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: token_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: token_embed.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: position_embed.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroup_count = self.num_tokens.div_ceil(self.workgroup_size);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use wgpu::util::DeviceExt;

    use super::*;

    fn cpu_embedding(token_ids: &[u32], token_embed: &[f32], position_embed: &[f32], hidden_size: usize) -> Vec<f32> {
        let num_tokens = token_ids.len();
        let mut output = vec![0.0; num_tokens * hidden_size];
        for i in 0..num_tokens {
            let token_id = token_ids[i] as usize;
            for j in 0..hidden_size {
                output[i * hidden_size + j] = token_embed[token_id * hidden_size + j] + position_embed[i * hidden_size + j];
            }
        }
        output
    }

    #[tokio::test]
    async fn test_embedding() {
        let (device, queue) = crate::init_gpu().await.unwrap();
        let hidden_size = 768;
        let num_tokens = 1024;
        let embedding = Embedding::new(&device, hidden_size, num_tokens);
        let mut rng = StdRng::seed_from_u64(42);

        let token_ids: Vec<u32> = (0..num_tokens)
            .map(|_| rng.random_range(0..10000))
            .collect();
        let token_embed: Vec<f32> = (0..10000 * hidden_size)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let position_embed: Vec<f32> = (0..num_tokens * hidden_size)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        
        let token_ids_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("token ids buffer"),
            contents: bytemuck::cast_slice(&token_ids),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let token_embed_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("token embed buffer"),
            contents: bytemuck::cast_slice(&token_embed),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let position_embed_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("position embed buffer"),
            contents: bytemuck::cast_slice(&position_embed),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output buffer"),
            size: (num_tokens * hidden_size * std::mem::size_of::<f32>() as u32) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("embedding command encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("embedding compute pass"),
                timestamp_writes: None,
            });
            embedding.dispatch(
                &token_ids_buffer, 
                &token_embed_buffer, 
                &position_embed_buffer, 
                &output_buffer, 
                &device, 
                &mut pass
            );
        }
        let result  = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("result buffer"),
            size: (num_tokens * hidden_size * std::mem::size_of::<f32>() as u32) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &result,
            0,
            (num_tokens * hidden_size * std::mem::size_of::<f32>() as u32) as wgpu::BufferAddress,
        );
        queue.submit(Some(encoder.finish()));
        let buffer_slice = result.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |result| {
            result.unwrap();
        });
        let _ = device.poll(wgpu::PollType::Wait);
        let data = buffer_slice.get_mapped_range();
        let gpu_result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        let cpu_result = cpu_embedding(&token_ids, &token_embed, &position_embed, hidden_size as usize);

        for i in 0..num_tokens as usize {
            for j in 0..hidden_size as usize {
                let idx = i * hidden_size as usize + j;
                assert!((gpu_result[idx] - cpu_result[idx]).abs() < 1e-4, "Mismatch at token {}, hidden {}", i, j);
            }
        }
    }
}