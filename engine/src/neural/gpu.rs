//! The GPU the networks run on: one wgpu instance and adapter for the whole program, and a
//! device for each network. Making an instance or looking for adapters on one thread while
//! another draws with Vulkan can crash the Vulkan loader (it loads and unloads drivers as it
//! looks), so a program that renders as well, like the editor, calls `adapter` before it
//! starts drawing, and networks loaded later on worker threads only ask for devices.

use std::sync::{Mutex, OnceLock};

static ADAPTER: OnceLock<Result<(wgpu::Instance, wgpu::Adapter), String>> = OnceLock::new();
/// Devices are asked for one at a time.
static DEVICES: Mutex<()> = Mutex::new(());

/// The adapter, found the first time it is asked for.
pub fn adapter() -> Result<&'static wgpu::Adapter, String> {
    let found = ADAPTER.get_or_init(|| {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| format!("no GPU adapter: {e}"))?;
        Ok((instance, adapter))
    });
    found.as_ref().map(|(_, a)| a).map_err(Clone::clone)
}

/// A device of its own for a network whose buffers reach `bytes`.
pub fn device(label: &str, bytes: u64) -> Result<(wgpu::Device, wgpu::Queue, String), String> {
    let adapter = adapter()?;
    let _one_at_a_time = DEVICES.lock().unwrap_or_else(|e| e.into_inner());
    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffer_binding_size = limits.max_storage_buffer_binding_size.max(bytes);
    limits.max_buffer_size = limits.max_buffer_size.max(bytes);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some(label),
        required_limits: limits,
        ..Default::default()
    }))
    .map_err(|e| format!("GPU device: {e}"))?;
    Ok((device, queue, adapter.get_info().name))
}
