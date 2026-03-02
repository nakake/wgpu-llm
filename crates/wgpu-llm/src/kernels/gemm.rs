use std::collections::HashMap;

use crate::template;

pub struct Gemm {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    m: u32,
    n: u32,
    k: u32,
}

impl Gemm {
    pub fn new(device: &wgpu::Device, m: u32, n: u32, k: u32) -> Self {
        let shader = include_str!("../../shaders/gemm.wgsl");
        let mut map = HashMap::new();
        map.insert("M", m.to_string());
        map.insert("N", n.to_string());
        map.insert("K", k.to_string());
        map.insert("WG_X", 16.to_string());
        map.insert("WG_Y", 16.to_string());
        let replace_shader = template::render(shader, &map);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemm shader"),
            source: wgpu::ShaderSource::Wgsl(replace_shader.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gemm bind group layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gemm pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("gemm pipeline layout"),
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
            m,
            n,
            k,
        }
    }

    pub fn dispatch(
        &self,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        c: &wgpu::Buffer,
        device: &wgpu::Device,
        pass: &mut wgpu::ComputePass<'_>,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemm bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c.as_entire_binding(),
                },
            ],
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let x = self.m.div_ceil(16);
        let y = self.n.div_ceil(16);
        pass.dispatch_workgroups(x, y, 1);
    }
}

#[cfg(test)]
mod tests {
    use wgpu::util::DeviceExt;

    use crate::init_gpu;

    use super::*;
    fn cpu_gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    #[tokio::test]
    async fn test_gemm() {
        let (device, queue) = init_gpu().await.unwrap();
        let m = 32;
        let n = 32;
        let k = 32;
        let a_data: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| i as f32).collect();
        let a_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("a buffer"),
            contents: bytemuck::cast_slice(&a_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("b buffer"),
            contents: bytemuck::cast_slice(&b_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let c_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("c buffer"),
            size: (m * n * std::mem::size_of::<f32>() as u32) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let gemm = Gemm::new(&device, m, n, k);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gemm command encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemm compute pass"),
                timestamp_writes: None,
            });
            gemm.dispatch(&a_buffer, &b_buffer, &c_buffer, &device, &mut pass);
        }

        let c_read_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("c read buffer"),
            size: (m * n * std::mem::size_of::<f32>() as u32) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            &c_buffer,
            0,
            &c_read_buffer,
            0,
            (m * n * std::mem::size_of::<f32>() as u32) as wgpu::BufferAddress,
        );
        queue.submit(Some(encoder.finish()));

        let buffer_slice = c_read_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |result| {
            result.unwrap();
        });

        let _ = device.poll(wgpu::PollType::Wait);

        let data = buffer_slice.get_mapped_range();
        let gpu_result: &[f32] = bytemuck::cast_slice(&data);

        let mut cpu_result = vec![0.0; (m * n) as usize];
        cpu_gemm(&a_data, &b_data, &mut cpu_result, m as usize, n as usize, k as usize);

        for i in 0..gpu_result.len() {
            assert!(
                (gpu_result[i] - cpu_result[i]).abs() < 1e-3,
                "Mismatch at index {}: gpu {} vs cpu {}",
                i,
                gpu_result[i],
                cpu_result[i]
            );
        }
    }
}
