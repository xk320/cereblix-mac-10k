//go:build darwin && arm64 && cgo

package main

/*
#include <pthread.h>

static void cereblix_set_user_interactive_qos(void) {
	pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
}
*/
import "C"
import "runtime"

func prepareMineWorkerThread() {
	runtime.LockOSThread()
	C.cereblix_set_user_interactive_qos()
}

