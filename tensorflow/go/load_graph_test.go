package tensorflow

import (
	_ "fmt"
	"testing"
)

func TestAddGraph(t *testing.T) {
	_, _, err := LoadGraph("sample_graphs/plus.pbtxt")
	if err != nil {
		t.Error("Failed to load add graph")
	}
}

func TestMatMulGraph(t *testing.T) {
	_, _, err := LoadGraph("sample_graphs/matmul.pbtxt")
	if err != nil {
		t.Error("Failed to load matmul graph")
	}
}

func TestLinRegGraph(t *testing.T) {
	_, _, err := LoadGraph("sample_graphs/linreg.pbtxt")
	if err != nil {
		t.Error("Failed to load mnist graph")
	}
}

func TestSimpleMNISTGraph(t *testing.T) {
	_, _, err := LoadGraph("sample_graphs/simple_nw.pbtxt")
	if err != nil {
		t.Error("Failed to load mnist graph")
	}
}

func TestMNISTGraph(t *testing.T) {
	_, _, err := LoadGraph("sample_graphs/mnist.pbtxt")
	if err != nil {
		t.Error("Failed to load mnist graph")
	}
}
