package tensorflow

// #include "tensorflow/c/c_api.h"
import "C"

import (
	"bytes"
	"fmt"
	pb "github.com/berset/tensorflow/tensorflow/go/pb/tensorflow/core/framework"
	"github.com/golang/protobuf/proto"
	"io/ioutil"
	"log"
	"strings"
)

func LoadGraph(graphFileName string) (*Graph, map[string]Output, error) {
	g := NewGraph()

	ns := make(map[string]Output)

	by, err := ioutil.ReadFile(graphFileName)
	if err != nil {
		return nil, nil, err
	}

	gd := &pb.GraphDef{}

	err = proto.UnmarshalText(string(by), gd)

	if err != nil {
		return nil, nil, err
	}

	type DelayedBuild struct {
		B        *opBuilder
		InputIdx int
		Node     *pb.NodeDef
	}

	var delayed []DelayedBuild

Node:
	for _, node := range gd.Node {
		b := newOpBuilder(g, node.Op, node.Name)
		for key, attr := range node.Attr {
			err := handleAttr(b, key, attr)
			if err != nil {
				log.Fatal(err)
			}
		}
		for i, input := range node.Input {
			// TODO figure out what these decoration mean?
			inpname := strings.Replace(input, ":1", "", 1)
			inpname2 := strings.Replace(input, "^", "", 1)

			if ns[input].Op == nil && ns[inpname].Op != nil {
				fmt.Printf("!")
				b.AddInput(ns[inpname])
			} else if ns[input].Op == nil && ns[inpname2].Op != nil {
				fmt.Printf("?")
				//b.AddInput(ns[inpname2])
			} else if ns[input].Op == nil {
				delayed = append(delayed, DelayedBuild{
					B:        b,
					InputIdx: i,
					Node:     node,
				})
				continue Node
			} else {
				fmt.Printf(".")
				b.AddInput(ns[input])
			}
		}
		op, err := b.Build()
		if err != nil {
			input := node.Input[0]
			inpname := strings.Replace(input, ":1", "", 1)
			inpname2 := strings.Replace(input, "^", "", 1)
			fmt.Println("")
			fmt.Println(node.Input[0])
			fmt.Println(ns[node.Input[0]])
			fmt.Println(ns[inpname])
			fmt.Println(ns[inpname2])
			fmt.Println(node.Input)
			fmt.Println("err")
			log.Panic(err)
			//return Output{}, err
		}
		ns[node.Name] = Output{op, 0}
	}

	fmt.Println("delayed")
	fmt.Println(len(delayed))
	fmt.Println("dadada\n\n--------------------------------------------------------------\n")

	var delayed2 []DelayedBuild

Node2:
	for _, de := range delayed {
		// TODO FIXME
		b := de.B
		node := de.Node
		for i, input := range node.Input {
			if i >= de.InputIdx {
				inpname := strings.Replace(input, ":1", "", 1)
				if ns[input].Op == nil &&
					ns[inpname].Op != nil {
					fmt.Println("can fix!!!!")
					fmt.Println("can fix!!!!")
					fmt.Println("can fix!!!!")
					fmt.Println("can fix!!!!")
					fmt.Println("can fix!!!!")
					fmt.Println("can fix!!!!")
					fmt.Println("can fix!!!!")
					fmt.Println("can fix!!!!")
					fmt.Println("can fix!!!!")
					b.AddInput(ns[inpname])
				} else if ns[input].Op == nil {
					delayed2 = append(delayed2, DelayedBuild{
						B:        b,
						InputIdx: i,
						Node:     node,
					})
					continue Node2
					b.AddInput(ns[input])
				}
			}
		}
		op, err := b.Build()
		if err != nil {
			fmt.Println("err")
			fmt.Println(err)
			//return Output{}, err
		}
		ns[node.Name] = Output{op, 0}
		delayed2 = append(delayed2, de)
	}

	//fmt.Println("delayed2")
	//fmt.Println(len(delayed2))
	//fmt.Println("dadada\n\n--------------------------------------------------------------\n")
	//for j := 0; j < len(delayed); j += 1 {
	//    fmt.Println(delayed[j].Node.Name)
	//    for _, inp := range(delayed[j].Node.Input) {
	//        fmt.Printf("i: %s\n", inp)
	//        fmt.Println(ns[inp])
	//        fmt.Println(ns["cross_entropy_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits"])
	//    }
	//}

	fmt.Println("del, del")
	fmt.Println(len(delayed))
	fmt.Println(len(delayed2))

	return g, ns, nil
}

func handleAttr(b *opBuilder, key string, m *pb.AttrValue) error {
	switch x := m.Value.(type) {
	case *pb.AttrValue_S:
		b.SetAttrString(key, string(m.GetS()))
	case *pb.AttrValue_I:
		b.SetAttrInt(key, int64(m.GetI()))
	case *pb.AttrValue_F:
		b.SetAttrFloat(key, float32(m.GetF()))
	case *pb.AttrValue_B:
		b.SetAttrBool(key, bool(m.GetB()))
	case *pb.AttrValue_Type:
		b.SetAttrType(key, DataType(m.GetType()))
	case *pb.AttrValue_Shape:
		shape := []int64{}
		for _, dim := range m.GetShape().GetDim() {
			shape = append(shape, dim.Size)
		}
		if len(shape) > 0 {
			b.SetAttrShape(key, shape)
		} else {
			shape = append(shape, -1)
			b.SetAttrShape(key, shape)
		}
	case *pb.AttrValue_Tensor:
		b.SetAttrTensor(key, t2t(m.GetTensor()))
	case *pb.AttrValue_List:
		//		b.EncodeVarint(1<<3 | proto.WireBytes)
		//		if err := b.EncodeMessage(x.List); err != nil {
		//			return err
		//		}
	case *pb.AttrValue_Func:
		//		b.EncodeVarint(10<<3 | proto.WireBytes)
		//		if err := b.EncodeMessage(x.Func); err != nil {
		//			return err
		//		}
	case *pb.AttrValue_Placeholder:
		//		b.EncodeVarint(9<<3 | proto.WireBytes)
		//		b.EncodeStringBytes(x.Placeholder)
	case nil:
	default:
		return fmt.Errorf("AttrValue.Value has unexpected type %T", x)
	}
	return nil
}

func t2t(in *pb.TensorProto) *Tensor {
	out := &Tensor{}
	out.dt = DataType(in.Dtype)
	out.shape = []int64{}
	for _, dim := range in.TensorShape.GetDim() {
		out.shape = append(out.shape, dim.Size)
	}
	out.buf = bytes.NewBuffer(in.TensorContent)
	return out
}
