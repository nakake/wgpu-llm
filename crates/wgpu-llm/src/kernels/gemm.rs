use std::collections::HashMap;

use wgpu::{BindGroupLayoutEntry, Device};

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
}