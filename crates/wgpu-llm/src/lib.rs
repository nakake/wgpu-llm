mod template;
mod kernels;
mod config;

use wgpu::{Features, InstanceDescriptor, Limits};

pub async fn init_gpu() -> anyhow::Result<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu-llm device"),
            required_features: Features::default(),
            required_limits: Limits::default(),
            ..Default::default()
        })
        .await?;

    Ok((device, queue))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_init_gpu() {
        let (device, queue) = init_gpu().await.unwrap();
        println!("{:?}", device.limits());
    }
}
