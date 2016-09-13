package tensorflow

// #include "tensorflow/c/c_api.h"
import "C"

import (
	"fmt"
    pb "github.com/berset/tensorflow/tensorflow/go/pb/tensorflow/core/framework"
)

func loadNodes() []pb.NodeDef {
	var nodes []pb.NodeDef
	nodes = append(nodes, pb.NodeDef{
		Name: "p1",
		Op:   "Placeholder",
		Attr: map[string]*pb.AttrValue{
            "dtype": &pb.AttrValue{
                Value: &pb.AttrValue_Type{
                    Type: pb.DataType_DT_INT64,
			    },
            },
		},
	})

	nodes = append(nodes, pb.NodeDef{
		Name: "p2",
		Op:   "Placeholder",
		Attr: map[string]*pb.AttrValue{
            "dtype": &pb.AttrValue{
                Value: &pb.AttrValue_Type{
                    Type: pb.DataType_DT_INT64,
			    },
            },
		},
	})

	nodes = append(nodes, pb.NodeDef{
		Name:  "neg1",
		Op:    "Add",
		Input: []string{"p1", "p2"},
	})

	return nodes
}

func LoadGraph(graphFileName string) (*Graph, map[string]Output, error) {
	g := NewGraph()

	ns := make(map[string]Output)

	fmt.Printf("about to load: %s\n", graphFileName)

	nodes := loadNodes()

	for _, node := range nodes {
		b := newOpBuilder(g, node.Op, node.Name)
		for _, attr := range node.Attr {
            handleAttr(b, attr)
		}
		for _, input := range node.Input {
			b.AddInput(ns[input])
		}
		op, err := b.Build()
		if err != nil {
			fmt.Println("err")
			fmt.Println(err)
			//return Output{}, err
		}
		ns[node.Name] = Output{op, 0}
	}

	return g, ns, nil
}

func handleAttr(b *opBuilder, m *pb.AttrValue) error {
	switch x := m.Value.(type) {
//	case *AttrValue_S:
//		b.EncodeVarint(2<<3 | proto.WireBytes)
//		b.EncodeRawBytes(x.S)
//	case *AttrValue_I:
//		b.EncodeVarint(3<<3 | proto.WireVarint)
//		b.EncodeVarint(uint64(x.I))
//	case *AttrValue_F:
//		b.EncodeVarint(4<<3 | proto.WireFixed32)
//		b.EncodeFixed32(uint64(math.Float32bits(x.F)))
//	case *AttrValue_B:
//		t := uint64(0)
//		if x.B {
//			t = 1
//		}
//		b.EncodeVarint(5<<3 | proto.WireVarint)
//		b.EncodeVarint(t)
	case *pb.AttrValue_Type:
		b.SetAttrType("dtype", DataType(m.GetType()))
//	case *AttrValue_Shape:
//		b.EncodeVarint(7<<3 | proto.WireBytes)
//		if err := b.EncodeMessage(x.Shape); err != nil {
//			return err
//		}
//	case *AttrValue_Tensor:
//		b.EncodeVarint(8<<3 | proto.WireBytes)
//		if err := b.EncodeMessage(x.Tensor); err != nil {
//			return err
//		}
//	case *AttrValue_List:
//		b.EncodeVarint(1<<3 | proto.WireBytes)
//		if err := b.EncodeMessage(x.List); err != nil {
//			return err
//		}
//	case *AttrValue_Func:
//		b.EncodeVarint(10<<3 | proto.WireBytes)
//		if err := b.EncodeMessage(x.Func); err != nil {
//			return err
//		}
//	case *AttrValue_Placeholder:
//		b.EncodeVarint(9<<3 | proto.WireBytes)
//		b.EncodeStringBytes(x.Placeholder)
	case nil:
	default:
		return fmt.Errorf("AttrValue.Value has unexpected type %T", x)
	}
	return nil
}
