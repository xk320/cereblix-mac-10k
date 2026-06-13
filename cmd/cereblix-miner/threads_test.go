package main

import "testing"

func TestRecommendedThreads(t *testing.T) {
	tests := []struct {
		name   string
		goos   string
		goarch string
		cpus   int
		want   int
	}{
		{name: "apple silicon overprovisions for throughput", goos: "darwin", goarch: "arm64", cpus: 10, want: 11},
		{name: "apple silicon rounds up", goos: "darwin", goarch: "arm64", cpus: 8, want: 9},
		{name: "intel mac keeps cpu count", goos: "darwin", goarch: "amd64", cpus: 8, want: 8},
		{name: "linux keeps cpu count", goos: "linux", goarch: "arm64", cpus: 16, want: 16},
		{name: "minimum one", goos: "darwin", goarch: "arm64", cpus: 0, want: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := recommendedThreadsFor(tt.goos, tt.goarch, tt.cpus); got != tt.want {
				t.Fatalf("recommendedThreadsFor(%q, %q, %d) = %d, want %d", tt.goos, tt.goarch, tt.cpus, got, tt.want)
			}
		})
	}
}
