use synfx_dsp_jit::engine::{CodeEngine, CodeEngineBackend};

use anyhow;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

fn main() {
    let mut engine = CodeEngine::new_stdlib();
    let backend = engine.get_backend();

    engine.set_debug(true);

    start_backend(backend, move || {
        use synfx_dsp_jit::build::*;

        let freqs = [440.0, 880.0, 220.0];
        let mut i = 0;
        loop {
            engine.query_returns();

            let freq = freqs[i];
            i = (i + 1) % freqs.len();

            engine.upload(stmts(&[assign(
                "&sig1",
                op_mul(literal(0.3), call("phase", 1, &[literal(freq)])),
            )]));
            println!("{}", engine.get_debug_info());

            std::thread::sleep(std::time::Duration::from_millis(300));
        }
    });
}

pub fn run<T, F: FnMut()>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut backend: CodeEngineBackend,
    mut frontend_loop: F,
) -> Result<(), anyhow::Error>
where
    T: cpal::Sample,
{
    let sample_rate = config.sample_rate.0 as f32;
    let channels = config.channels as usize;

    backend.set_sample_rate(sample_rate);

    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);
    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            backend.process_updates();

            let mut out_iter = data.chunks_mut(channels);
            while let Some(frame) = out_iter.next() {
                let (sig1, _sig2, _ret) = backend.process(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

                for sample in frame.iter_mut() {
                    let value: T = cpal::Sample::from::<f32>(&sig1);
                    *sample = value;
                }
            }
        },
        err_fn,
    )?;
    stream.play()?;

    frontend_loop();

    Ok(())
}

// This function starts the CPAL backend and
// runs the audio loop with the NodeExecutor.
fn start_backend<F: FnMut()>(backend: CodeEngineBackend, frontend_loop: F) {
    let host = cpal::default_host();
    let device = host.default_output_device().expect("Finding useable audio device");
    let config = device.default_output_config().expect("A workable output config");

    match config.sample_format() {
        cpal::SampleFormat::F32 => run::<f32, F>(&device, &config.into(), backend, frontend_loop),
        cpal::SampleFormat::I16 => run::<i16, F>(&device, &config.into(), backend, frontend_loop),
        cpal::SampleFormat::U16 => run::<u16, F>(&device, &config.into(), backend, frontend_loop),
    }
    .expect("cpal works fine");
}
