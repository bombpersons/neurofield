//! GPU time by kernel, for finding what's slow (`GAME_DREAM_PROFILE=1`): on every so many
//! frames, timestamps between the frame's dispatches and passes, read back without waiting, and
//! every few of those the mean milliseconds of each label printed, most first.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Timestamps a frame at most.
const STAMPS: u32 = 1024;
/// A frame timed every this many.
const EVERY: u32 = 15;
/// Timed frames between printouts.
const PRINT: u32 = 20;

pub(crate) struct Profiler {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    /// What ended at each timestamp after the first, this frame's.
    labels: Vec<String>,
    timing: bool,
    /// The labels of the frame being read back, and whether its mapping's done.
    pending: Option<(Vec<String>, Arc<AtomicBool>, bool)>,
    totals: HashMap<String, (f64, u32)>,
    period: f32,
    frame: u32,
    timed: u32,
    every: u32,
    /// Labels dispatches with their sizes (PROFILE_SHAPES set).
    pub by_shape: bool,
}

impl Profiler {
    /// None unless asked for and the device can time inside passes.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Profiler> {
        // 1 for the usual spacing, or a frame timed every so many (2 or more).
        let every = std::env::var("GAME_DREAM_PROFILE").ok().filter(|v| v != "0")?.parse().ok().filter(|&n: &u32| n > 1).unwrap_or(EVERY);
        let need = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        if !device.features().contains(need) {
            eprintln!("dream profile: the device can't time inside passes ({:?})", device.features() & need);
            return None;
        }
        let size = STAMPS as u64 * 8;
        Some(Profiler {
            set: device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("dream profile"), ty: wgpu::QueryType::Timestamp, count: STAMPS }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor { label: Some("dream profile"), size, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }),
            read: device.create_buffer(&wgpu::BufferDescriptor { label: Some("dream profile read"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
            labels: Vec::new(),
            timing: false,
            pending: None,
            totals: HashMap::new(),
            period: queue.get_timestamp_period(),
            frame: 0,
            timed: 0,
            every,
            by_shape: std::env::var_os("PROFILE_SHAPES").is_some(),
        })
    }

    /// A frame begins: timed if it's its turn and the last timed one's been read.
    pub fn begin(&mut self, enc: &mut wgpu::CommandEncoder) {
        self.collect();
        self.frame += 1;
        self.timing = self.pending.is_none() && self.frame.is_multiple_of(self.every);
        self.labels.clear();
        if self.timing {
            enc.write_timestamp(&self.set, 0);
        }
    }

    /// The next stamp's index, `label` ending there; None if not timing.
    fn next(&mut self, label: &str) -> Option<u32> {
        if !self.timing || self.labels.len() as u32 + 1 >= STAMPS {
            return None;
        }
        self.labels.push(label.to_string());
        Some(self.labels.len() as u32)
    }

    pub fn stamp(&mut self, enc: &mut wgpu::CommandEncoder, label: &str) {
        if let Some(i) = self.next(label) {
            enc.write_timestamp(&self.set, i);
        }
    }

    pub fn stamp_pass(&mut self, pass: &mut wgpu::ComputePass, label: &str) {
        if let Some(i) = self.next(label) {
            pass.write_timestamp(&self.set, i);
        }
    }

    /// The frame's stamps resolved and copied for reading.
    pub fn end(&mut self, enc: &mut wgpu::CommandEncoder) {
        if !self.timing {
            return;
        }
        let n = self.labels.len() as u32 + 1;
        enc.resolve_query_set(&self.set, 0..n, &self.resolve, 0);
        enc.copy_buffer_to_buffer(&self.resolve, 0, &self.read, 0, n as u64 * 8);
        self.pending = Some((std::mem::take(&mut self.labels), Arc::new(AtomicBool::new(false)), false));
        self.timing = false;
    }

    /// The last timed frame's stamps, once they've reached the CPU.
    fn collect(&mut self) {
        let Some((labels, done, asked)) = &mut self.pending else { return };
        if !*asked {
            *asked = true;
            let done = done.clone();
            self.read.slice(..).map_async(wgpu::MapMode::Read, move |r| done.store(r.is_ok(), Ordering::Release));
            return;
        }
        if !done.load(Ordering::Acquire) {
            return;
        }
        let labels = std::mem::take(labels);
        {
            let Ok(view) = self.read.slice(..(labels.len() as u64 + 1) * 8).get_mapped_range() else { return };
            let t: &[u64] = bytemuck::cast_slice(&view);
            for (i, l) in labels.iter().enumerate() {
                let ms = t[i + 1].saturating_sub(t[i]) as f64 * self.period as f64 / 1e6;
                let e = self.totals.entry(l.clone()).or_default();
                e.0 += ms;
                e.1 += 1;
            }
        }
        self.read.unmap();
        self.pending = None;
        self.timed += 1;
        if self.timed.is_multiple_of(PRINT) {
            let mut rows: Vec<(String, f64, u32)> = self.totals.drain().map(|(l, (ms, n))| (l, ms / PRINT as f64, n / PRINT)).collect();
            rows.sort_by(|a, b| b.1.total_cmp(&a.1));
            let total: f64 = rows.iter().map(|r| r.1).sum();
            eprintln!("dream profile: {total:.2} ms a frame on the GPU");
            for (l, ms, n) in rows.iter().take(if self.by_shape { 1000 } else { 30 }) {
                eprintln!("  {ms:7.3} ms  {l} (x{n})");
            }
        }
    }
}
