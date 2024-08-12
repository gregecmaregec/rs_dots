use bytemuck::{Pod, Zeroable};
use rand::Rng;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

const TARGET_WIDTH: u32 = 861;
const TARGET_HEIGHT: u32 = 315;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 3],
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    particles: Vec<Particle>,
    mouse: Mouse,
    start_time: Instant,
}

#[derive(Clone, Copy)]
struct Mouse {
    x: f32,
    y: f32,
    radius: f32,
}

#[derive(Clone)]
struct Particle {
    x: f32,
    y: f32,
    size: f32,
    speed_x: f32,
    speed_y: f32,
    color: [f32; 3],
    lifespan: i32,
}

impl Particle {
    fn new(x: f32, y: f32) -> Self {
        let mut rng = rand::thread_rng();
        Particle {
            x,
            y,
            size: rng.gen_range(1.0..4.0),
            speed_x: (rng.gen::<f32>() - 0.2) * 0.1,
            speed_y: (rng.gen::<f32>() - 0.2) * 0.1,
            color: [
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
                rng.gen_range(0.0..1.0),
            ],
            lifespan: rng.gen_range(300..500),
        }
    }

    fn update(&mut self) {
        self.x += self.speed_x;
        self.y += self.speed_y;
        self.lifespan -= 1;

        if self.size > 0.2 && self.lifespan < 100 {
            self.size -= 0.02;
        }
    }
}

impl State {
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });
        let surface = unsafe { instance.create_surface(&window) }.unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_supported_formats(&adapter)[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x3],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: (4 * std::mem::size_of::<Vertex>() * 1000) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Index Buffer"),
            size: (6 * std::mem::size_of::<u16>() * 1000) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            particles: Vec::new(),
            mouse: Mouse {
                x: 0.0,
                y: 0.0,
                radius: 150.0,
            },
            start_time: Instant::now(),
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let scale_x = TARGET_WIDTH as f32 / self.size.width as f32;
                let scale_y = TARGET_HEIGHT as f32 / self.size.height as f32;
                self.mouse.x = position.x as f32 * scale_x;
                self.mouse.y = position.y as f32 * scale_y;

                // Create particles on mouse move
                for _ in 0..1 {
                    self.particles.push(Particle::new(self.mouse.x, self.mouse.y));
                }
                true
            }
            _ => false,
        }
    }

    fn update(&mut self) {
        self.create_particles();
        self.handle_particles();
    }

    fn create_particles(&mut self) {
        if self.particles.len() < 75 {
            let mut rng = rand::thread_rng();
            self.particles.push(Particle::new(
                rng.gen_range(0.0..TARGET_WIDTH as f32),
                rng.gen_range(0.0..TARGET_HEIGHT as f32),
            ));
        }
    }

    fn handle_particles(&mut self) {
        self.particles.retain_mut(|p| {
            p.update();
            p.size > 0.2 && p.lifespan > 0
        });
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut index = 0;

        // Render particles
        for particle in &self.particles {
            let color = particle.color;
            let x = particle.x / TARGET_WIDTH as f32 * 2.0 - 1.0;
            let y = -(particle.y / TARGET_HEIGHT as f32 * 2.0 - 1.0);
            let size = particle.size / TARGET_HEIGHT as f32;

            vertices.extend_from_slice(&[
                Vertex {
                    position: [x - size, y - size],
                    color,
                },
                Vertex {
                    position: [x + size, y - size],
                    color,
                },
                Vertex {
                    position: [x - size, y + size],
                    color,
                },
                Vertex {
                    position: [x + size, y + size],
                    color,
                },
            ]);

            indices.extend_from_slice(&[
                index,
                index + 1,
                index + 2,
                index + 1,
                index + 3,
                index + 2,
            ]);
            index += 4;
        }

        // Render connections
        for i in 0..self.particles.len() {
            for j in i + 1..self.particles.len() {
                let dx = self.particles[i].x - self.particles[j].x;
                let dy = self.particles[i].y - self.particles[j].y;
                let distance = (dx * dx + dy * dy).sqrt();

                if distance < 100.0 {
                    let color = self.particles[i].color;
                    let x1 = self.particles[i].x / TARGET_WIDTH as f32 * 2.0 - 1.0;
                    let y1 = -(self.particles[i].y / TARGET_HEIGHT as f32 * 2.0 - 1.0);
                    let x2 = self.particles[j].x / TARGET_WIDTH as f32 * 2.0 - 1.0;
                    let y2 = -(self.particles[j].y / TARGET_HEIGHT as f32 * 2.0 - 1.0);

                    vertices.extend_from_slice(&[
                        Vertex {
                            position: [x1, y1],
                            color,
                        },
                        Vertex {
                            position: [x2, y2],
                            color,
                        },
                    ]);

                    indices.extend_from_slice(&[index, index + 1]);
                    index += 2;
                }
            }
        }

        self.queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(&vertices),
        );
        self.queue
            .write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

// ... (previous code remains the same)

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Crystal Nature Simulation")
        .with_inner_size(winit::dpi::PhysicalSize::new(TARGET_WIDTH, TARGET_HEIGHT))
        .build(&event_loop)
        .unwrap();

    let mut state = pollster::block_on(State::new(&window));

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => {
            if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
        }
        Event::RedrawRequested(window_id) if window_id == window.id() => {
            state.update();
            match state.render() {
                Ok(_) => {}
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        _ => {}
    });
}