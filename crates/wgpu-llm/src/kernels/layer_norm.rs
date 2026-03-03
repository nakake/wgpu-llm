use std::collections::HashMap;

use crate::template;

pub struct LayerNorm {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    hidden_size: u32,
    num_tokens: u32,
    workgroup_size: u32,
}

impl LayerNorm {
    pub fn new(device: &wgpu::Device, hidden_size: u32, num_tokens: u32) -> Self {
        let shader = include_str!("../../shaders/layer_norm.wgsl");
        let mut map = HashMap::new();
        let workgroup_size = 256;
        map.insert("HIDDEN_SIZE", hidden_size.to_string());
        map.insert("NUM_TOKENS", num_tokens.to_string());
        map.insert("WG_SIZE", workgroup_size.to_string());
        let replace_shader = template::render(shader, &map);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("layer norm shader"),
            source: wgpu::ShaderSource::Wgsl(replace_shader.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layer norm bind group layout"),
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
            label: Some("layer norm pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("layer norm pipeline layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                }),
            ),
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
            workgroup_size,
        }
    }

    pub fn dispatch(
      &self,
      input: &wgpu::Buffer,
      gamma: &wgpu::Buffer,
      beta: &wgpu::Buffer,
      output: &wgpu::Buffer,
      device: &wgpu::Device,
      pass: &mut wgpu::ComputePass<'_>,
    ) {
      let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
          label: Some("layer norm bind group"),
          layout: &self.bind_group_layout,
          entries: &[
              wgpu::BindGroupEntry {
                  binding: 0,
                  resource: input.as_entire_binding(),
              },
              wgpu::BindGroupEntry {
                  binding: 1,
                  resource: gamma.as_entire_binding(),
              },
              wgpu::BindGroupEntry {
                  binding: 2,
                  resource: beta.as_entire_binding(),
              },
              wgpu::BindGroupEntry {
                  binding: 3,
                  resource: output.as_entire_binding(),
              },
          ],
      });

      pass.set_pipeline(&self.pipeline);
      pass.set_bind_group(0, &bind_group, &[]);
      let x = self.num_tokens.div_ceil(self.workgroup_size);
      pass.dispatch_workgroups(x, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use wgpu::util::DeviceExt;

    fn cpu_layer_norm(input: &[f32], gamma: &[f32], beta: &[f32], hidden_size: usize) -> Vec<f32> {
        let num_tokens = input.len() / hidden_size;
        let mut output = vec![0.0; input.len()];

        for i in 0..num_tokens {
            let start = i * hidden_size;
            let end = start + hidden_size;
            let token_input = &input[start..end];

            let mean: f32 = token_input.iter().sum::<f32>() / hidden_size as f32;
            let variance: f32 = token_input
                .iter()
                .map(|x| (x - mean)
                .powi(2))
                .sum::<f32>() / hidden_size as f32;

            for j in 0..hidden_size {
                output[start + j] = (token_input[j] - mean) / (variance + 1e-5).sqrt() * gamma[j] + beta[j];
            }
        }

        output
    }

    #[tokio::test]
    async fn test_layer_norm() {
        let (device, queue) = crate::init_gpu().await.unwrap();

        let hidden_size = 512;
        let num_tokens = 1024;
        let mut rng = StdRng::seed_from_u64(42);

        let layer_norm = LayerNorm::new(&device, hidden_size, num_tokens);
        let input_data: Vec<f32> = (0..num_tokens * hidden_size)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let gamma_data: Vec<f32> = (0..hidden_size)
            .map(|_| rng.random_range(0.5..1.5))
            .collect();
        let beta_data: Vec<f32> = (0..hidden_size)
            .map(|_| rng.random_range(-0.5..0.5))
            .collect();

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input buffer"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let gamma_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gamma buffer"),
            contents: bytemuck::cast_slice(&gamma_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let beta_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("beta buffer"),
            contents: bytemuck::cast_slice(&beta_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output buffer"),
            size: (num_tokens * hidden_size * std::mem::size_of::<f32>() as u32) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("layer norm command encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("layer norm compute pass"),
                timestamp_writes: None,
            });
            layer_norm.dispatch(&input_buffer, &gamma_buffer, &beta_buffer, &output_buffer, &device, &mut pass);
        }
        let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("result buffer"),
            size: (num_tokens * hidden_size * std::mem::size_of::<f32>() as u32) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &result_buffer,
            0,
            (num_tokens * hidden_size * std::mem::size_of::<f32>() as u32) as wgpu::BufferAddress,
        );
        queue.submit(Some(encoder.finish()));

        let buffer_slice = result_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |result| {
            result.unwrap();
        });
        
        let _ = device.poll(wgpu::PollType::Wait);
        
        let data = buffer_slice.get_mapped_range();
        let gpu_result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        let cpu_result = cpu_layer_norm(&input_data, &gamma_data, &beta_data, hidden_size as usize);

        for (gpu, cpu) in gpu_result.iter().zip(cpu_result.iter()) {
            assert!((gpu - cpu).abs() < 1e-5);
        }
    }
}
