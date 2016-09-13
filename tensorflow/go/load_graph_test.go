package tensorflow

import (
	"fmt"
	"testing"
)

func TestCPPLoadGraph(t *testing.T) {
	fmt.Println("here!")
	sess, err := NewSessionGraph()
	fmt.Println(sess)
	fmt.Println(err)

	img := []float32{}
	for i := 0; i < 784; i++ {
		img = append(img, 1)
	}
	t1, _ := NewTensor(img)

	inputs := make(map[Output]*Tensor)
	inputs[ns["x"]] = t1
	//inputs[ns["Const1"]] = t2

	fmt.Println("outpu")
	fmt.Println(ns["output_layer/W"])
	outputs := []Output{ns["output_layer/W"]}

	output, err := s.Run(inputs, outputs, []*Operation{ns["output_layer/W"].Op})

	if err == nil {
		//fmt.Printf("shape: ")
		//fmt.Println(output[0].Shape())
		fmt.Printf("value: ")
		fmt.Println(output[0].Value())
	} else {
		log.Panic(err)
	}
}

//func TestAddGraph(t *testing.T) {
//    _, _, err := LoadGraph("sample_graphs/plus.pbtxt")
//    if err != nil {
//        t.Error("Failed to load add graph")
//    }
//}
//
//func TestMatMulGraph(t *testing.T) {
//    _, _, err := LoadGraph("sample_graphs/matmul.pbtxt")
//    if err != nil {
//        t.Error("Failed to load matmul graph")
//    }
//}

//func TestMNISTGraph(t *testing.T) {
//    _, _, err := LoadGraph("sample_graphs/simple_nw.pbtxt")
//    if err != nil {
//        t.Error("Failed to load mnist graph")
//    }
//}

//func TestMNISTGraph(t *testing.T) {
//    _, _, err := LoadGraph("sample_graphs/mnist.pbtxt")
//    if err != nil {
//        t.Error("Failed to load mnist graph")
//    }
//}
