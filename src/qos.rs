//! Thread QoS. On macOS this is libSystem's `pthread_set_qos_class_self_np`.
//! Elsewhere it is a no-op so the semantic tests can build on Linux CI;
//! nothing here changes the macOS calls.

#[cfg(target_os = "macos")]
pub use libc::pthread_set_qos_class_self_np;
#[cfg(target_os = "macos")]
pub use libc::qos_class_t;

#[cfg(not(target_os = "macos"))]
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
#[repr(u32)]
pub enum qos_class_t {
    QOS_CLASS_USER_INTERACTIVE = 0x21,
    QOS_CLASS_USER_INITIATED = 0x19,
    QOS_CLASS_UTILITY = 0x11,
}

#[cfg(not(target_os = "macos"))]
pub unsafe fn pthread_set_qos_class_self_np(_class: qos_class_t, _relative: i32) -> i32 {
    0
}
