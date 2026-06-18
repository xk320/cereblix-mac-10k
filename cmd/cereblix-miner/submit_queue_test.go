package main

import "testing"

func TestSubmitQueueSize(t *testing.T) {
	tests := []struct {
		name    string
		threads int
		want    int
	}{
		{name: "minimum capacity", threads: 1, want: 16},
		{name: "zero threads still gets minimum capacity", threads: 0, want: 16},
		{name: "scales with larger thread counts", threads: 10, want: 40},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := submitQueueSize(tt.threads); got != tt.want {
				t.Fatalf("submitQueueSize(%d) = %d, want %d", tt.threads, got, tt.want)
			}
		})
	}
}

func TestEnqueueSubmitFallsBackWhenQueueIsFull(t *testing.T) {
	queue := make(chan submitReq, 1)
	queue <- submitReq{id: "queued", nonce: 1, height: 2}

	req := submitReq{id: "fallback", nonce: 3, height: 4}
	var got submitReq
	called := false
	queued := enqueueSubmit(queue, req, func(req submitReq) {
		called = true
		got = req
	})

	if queued {
		t.Fatal("enqueueSubmit reported queued for a full queue")
	}
	if !called {
		t.Fatal("enqueueSubmit did not call fallback for a full queue")
	}
	if got != req {
		t.Fatalf("fallback got %+v, want %+v", got, req)
	}
}
