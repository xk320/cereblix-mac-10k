package main

import (
	"encoding/hex"
	"testing"
)

func must32(t *testing.T, s string) [32]byte {
	t.Helper()
	raw, err := hex.DecodeString(s)
	if err != nil || len(raw) != 32 {
		t.Fatalf("bad 32-byte hex %q", s)
	}
	var out [32]byte
	copy(out[:], raw)
	return out
}

func mustBytes(t *testing.T, s string) []byte {
	t.Helper()
	raw, err := hex.DecodeString(s)
	if err != nil {
		t.Fatalf("bad hex %q", s)
	}
	return raw
}

func TestHashMeetsTargetBytes(t *testing.T) {
	target := mustBytes(t, "0000001802dd529101cd784fd421f6859562541352ace37742b95cd8370d68eb")

	if !hashMeetsTargetBytes(must32(t, "0000001802dd529101cd784fd421f6859562541352ace37742b95cd8370d68eb"), target) {
		t.Fatal("equal hash should meet target")
	}
	if !hashMeetsTargetBytes(must32(t, "0000001802dd529101cd784fd421f6859562541352ace37742b95cd8370d68ea"), target) {
		t.Fatal("lower hash should meet target")
	}
	if hashMeetsTargetBytes(must32(t, "0000001802dd529101cd784fd421f6859562541352ace37742b95cd8370d68ec"), target) {
		t.Fatal("higher hash should not meet target")
	}
	if hashMeetsTargetBytes(must32(t, "0100000000000000000000000000000000000000000000000000000000000000"), target) {
		t.Fatal("much higher hash should not meet target")
	}
}
