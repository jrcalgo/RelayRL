use once_cell::sync::Lazy;
use std::convert::TryInto;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};
use tokio::runtime::Runtime as TokioRuntime;
use tokio::runtime::{Builder as TokioBuilder, Runtime};

#[cfg(feature = "console-subscriber")]
static CONSOLE_SUBSCRIBER_STATUS: Lazy<Arc<RwLock<AtomicBool>>> =
    Lazy::new(|| Arc::new(RwLock::new(AtomicBool::new(false))));

#[cfg(feature = "console-subscriber")]
pub(crate) fn register_console_subscriber_status(status: bool) {
    let mut registery = CONSOLE_SUBSCRIBER_STATUS
        .write()
        .expect("Console Subscriber Status unavailable");
    *registery = AtomicBool::from(status);
}

#[cfg(feature = "console-subscriber")]
pub fn get_or_init_console_subscriber() {
    {
        let status: &bool = unsafe {
            &*CONSOLE_SUBSCRIBER_STATUS
                .read()
                .expect("Console Subscriber Status unavailable")
                .as_ptr()
                .cast_const()
        };
        if *status {
            return;
        } else {
            console_subscriber::init();
            register_console_subscriber_status(true);
        }
    }
}

static GLOBAL_TOKIO_RUNTIME: Lazy<RwLock<Option<Arc<TokioRuntime>>>> =
    Lazy::new(|| RwLock::new(None));

pub(crate) fn register_tokio_runtime(runtime: Arc<TokioRuntime>) {
    let mut registery = GLOBAL_TOKIO_RUNTIME
        .write()
        .expect("Global Tokio Runtime unavailable");
    *registery = Some(runtime);
}

pub fn get_or_init_tokio_runtime() -> Arc<TokioRuntime> {
    // Assuming there is a current runtime available
    {
        let some: bool = GLOBAL_TOKIO_RUNTIME
            .read()
            .expect("GLOBAL_TOKIO_RUNTIME not initialized")
            .is_some();
        if some {
            return GLOBAL_TOKIO_RUNTIME
                .read()
                .expect("GLOBAL_TOKIO_RUNTIME not initialized")
                .clone()
                .expect("GLOBAL_TOKIO_RUNTIME not initialized");
        }
    }

    // get core count of the machine
    let core_count: i32 = std::thread::available_parallelism()
        .unwrap_or_else(|_| std::num::NonZeroUsize::new(1).unwrap())
        .get() as i32;
    // if core count is 1, use single threaded runtime
    // use 1 less than core count to account for main thread
    let worker_threads: usize = (core_count - 1)
        .try_into()
        .expect("Unable to convert core count to usize");

    // Otherwise create new runtime
    let runtime_multi_thread: Arc<Runtime> = Arc::new(
        TokioBuilder::new_multi_thread()
            .worker_threads(worker_threads)
            .thread_name("relayrl::tokio_worker")
            .thread_stack_size(10 * 1024 * 1024)
            .enable_all()
            .build()
            .expect("Unable to create tokio runtime"),
    );
    register_tokio_runtime(runtime_multi_thread.clone());
    runtime_multi_thread
}
