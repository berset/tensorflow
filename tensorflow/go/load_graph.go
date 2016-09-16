package tensorflow

// #include "tensorflow/c/c_api.h"
import "C"

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	pb "github.com/berset/tensorflow/tensorflow/go/pb/tensorflow/core/framework"
	"github.com/golang/protobuf/proto"
	"io/ioutil"
	"log"
	"strings"
)

type NetworkParams []struct {
	Name   string `json:"name"`
	Matrix json.RawMessage
}

type FloatMatrix1D []float32
type FloatMatrix2D [][]float32
type Int64Matrix1D []int64
type Int64Matrix2D [][]int64

func InitWeights(g *Graph, ns map[string]Output, jsonfile string) ([]*Operation, error) {
	jsonBlob, err := ioutil.ReadFile(jsonfile)
	if err != nil {
		return nil, err
	}
	var nwParams NetworkParams
	err = json.Unmarshal(jsonBlob, &nwParams)
	if err != nil {
		fmt.Println("error:", err)
		return nil, err
	}
    var ops []*Operation
	for _, nwParam := range nwParams {
		f1 := FloatMatrix1D{}
		f2 := FloatMatrix2D{}
        realName := strings.Replace(nwParam.Name, ":0", "", 1)
		fmt.Println(realName)
        constName := fmt.Sprintf("%s/init/Const", realName)
		constOp := newOpBuilder(g, "Const", constName)
		if err := json.Unmarshal(nwParam.Matrix, &f1); err == nil {
			constOp.SetAttrType("dtype", DataType(pb.DataType_DT_FLOAT))
	        buf := new(bytes.Buffer)
			for i := 0; i < len(f1); i++ {
				err = binary.Write(buf, nativeEndian, f1[i])
			}
            t := &Tensor{
                buf: buf,
                dt: DataType(pb.DataType_DT_FLOAT),
                shape: []int64{int64(len(f1))},
                }
			constOp.SetAttrTensor("value", t)
            op, err := constOp.Build()
            fmt.Println(err)
            ns[constName] = Output{op, 0}
			fmt.Println("f1")
		} else if err := json.Unmarshal(nwParam.Matrix, &f2); err == nil {
			constOp.SetAttrType("dtype", DataType(pb.DataType_DT_FLOAT))
	        buf := new(bytes.Buffer)
			for i := 0; i < len(f2); i++ {
			    for j := 0; j < len(f2[i]); j++ {
				    err = binary.Write(buf, nativeEndian, f2[i][j])
                }
			}
            t := &Tensor{
                buf: buf,
                dt: DataType(pb.DataType_DT_FLOAT),
                shape: []int64{int64(len(f2)), int64(len(f2[0]))},
                }
			constOp.SetAttrTensor("value", t)
            op, err := constOp.Build()
            fmt.Println(err)
            ns[constName] = Output{op, 0}
			fmt.Println("f2")
		}
		assignOp := newOpBuilder(g, "Assign", fmt.Sprintf("%s/init/Assign", realName))
        assignOp.AddInput(ns[realName])
        assignOp.AddInput(ns[constName])
        asop, err := assignOp.Build()
        fmt.Println(err)
        ops = append(ops, asop)
	}
	return ops, nil
}

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

	for _, node := range gd.Node {
		b := newOpBuilder(g, node.Op, node.Name)
		attr_map := make(map[string]*pb.AttrValue)
		for key, attr := range node.Attr {
			attr_map[key] = attr
			var err error
			err = handleAttr(node, b, key, attr)
			if err != nil {
				log.Panic(err)
			}
		}
		if attr_map["N"] != nil {
			n := int(attr_map["N"].GetI())
			var inputs []Output
			for _, input := range node.Input {
				inputs = append(inputs, ns[input])
				if len(inputs)%n == 0 {
					b.AddInputList(inputs, n)
					inputs = []Output{}
				}
			}
		} else {
			for _, input := range node.Input {
				// TODO figure out what these decoration mean?
				inpname := strings.Replace(input, ":1", "", 1)
				inpname2 := strings.Replace(input, "^", "", 1)

				if ns[input].Op == nil && ns[inpname].Op != nil {
					//fmt.Printf("!")
					b.AddInput(ns[inpname])
				} else if ns[input].Op == nil && ns[inpname2].Op != nil {
					// "^" seems to mean input less
					//fmt.Printf("?")
					//b.AddInput(ns[inpname2])
				} else if ns[input].Op != nil {
					//fmt.Printf(".")
					b.AddInput(ns[input])
				} else {
					return nil, nil, fmt.Errorf("input not found: ", input)
				}
			}
		}
		op, err := b.Build()
		if err != nil {
			fmt.Println(err)
			input := node.Input[0]
			inpname := strings.Replace(input, ":1", "", 1)
			inpname2 := strings.Replace(input, "^", "", 1)
			fmt.Println("ERRORROZZ")
			fmt.Println(node.Name)
			fmt.Println("--------------------")
			fmt.Println("attrs:")
			for key, attr := range node.Attr {
				fmt.Println(key)
				fmt.Println(attr)
			}
			fmt.Println("--------------------")
			fmt.Println(node.Input[0])
			fmt.Println("inp0")
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

	return g, ns, nil
}

func handleAttr(node *pb.NodeDef, b *opBuilder, key string, m *pb.AttrValue) error {
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
		dims := m.GetShape().GetDim()
		for _, dim := range dims {
			shape = append(shape, dim.Size)
		}
		b.SetAttrShape(key, shape)
	case *pb.AttrValue_Tensor:
		tensor, err := t2t(node, key, m.GetTensor())
		if err != nil {
			return err
		}
		b.SetAttrTensor(key, tensor)
	case *pb.AttrValue_List:
	case *pb.AttrValue_Func:
	case *pb.AttrValue_Placeholder:
		// TODO
		return fmt.Errorf("TODO - left to implement: %T", x)
	case nil:
	default:
		return fmt.Errorf("AttrValue.Value has unexpected type %T", x)
	}
	return nil
}

func t2t(node *pb.NodeDef, key string, in *pb.TensorProto) (*Tensor, error) {
	out := &Tensor{}
	out.dt = DataType(in.Dtype)
	out.shape = []int64{}
	for _, dim := range in.TensorShape.GetDim() {
		out.shape = append(out.shape, dim.Size)
	}
	buf := new(bytes.Buffer)
	var err error
	switch in.Dtype {
	case pb.DataType_DT_FLOAT:
		if len(out.shape) > 0 {
			prod := int64(1)
			for _, s := range out.shape {
				prod = prod * s
			}
			for i := int64(0); i < prod; i++ {
				err = binary.Write(buf, nativeEndian, in.FloatVal)
			}
		} else {
			err = binary.Write(buf, nativeEndian, in.FloatVal)
		}
	case pb.DataType_DT_DOUBLE:
		err = binary.Write(buf, nativeEndian, in.DoubleVal)

	case pb.DataType_DT_INT32:
	case pb.DataType_DT_UINT8:
	case pb.DataType_DT_INT16:
	case pb.DataType_DT_INT8:
		err = binary.Write(buf, nativeEndian, in.IntVal)

	case pb.DataType_DT_STRING:
		err = binary.Write(buf, nativeEndian, in.StringVal)
	case pb.DataType_DT_INT64:
		err = binary.Write(buf, nativeEndian, in.Int64Val)
	case pb.DataType_DT_BOOL:
		err = binary.Write(buf, nativeEndian, in.BoolVal)
	case pb.DataType_DT_HALF:
	case pb.DataType_DT_COMPLEX64:
	case pb.DataType_DT_QINT8:
	case pb.DataType_DT_QUINT8:
	case pb.DataType_DT_QINT32:
	case pb.DataType_DT_BFLOAT16:
	case pb.DataType_DT_QINT16:
	case pb.DataType_DT_QUINT16:
	case pb.DataType_DT_UINT16:
	case pb.DataType_DT_COMPLEX128:
	default:
		err = fmt.Errorf("TODO")
	}
	if err != nil {
		return nil, err
	}
	out.buf = buf
	return out, nil
}
