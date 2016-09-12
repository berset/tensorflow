package tensorflow

import (
	"fmt"
)

type Node struct {
	Name   string
	Op     string
	Inputs []string
	Attrs  []Attr
}

type Attr struct {
	Key   string
	Value interface{}
}

func loadNodes() []Node {
	t1, _ := NewTensor(int64(1))
	dt := t1.DataType()

	var nodes []Node
	nodes = append(nodes, Node{
		Name: "p1",
		Op:   "Placeholder",
		Attrs: []Attr{
			Attr{
				Key:   "dtype",
				Value: dt,
			},
		},
	})

	nodes = append(nodes, Node{
		Name: "p2",
		Op:   "Placeholder",
		Attrs: []Attr{
			Attr{
				Key:   "dtype",
				Value: dt,
			},
		},
	})

	nodes = append(nodes, Node{
		Name:   "neg1",
		Op:     "Add",
		Inputs: []string{"p1", "p2"},
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
		for _, attr := range node.Attrs {
			if attr.Key == "dtype" {
				b.SetAttrType("dtype", attr.Value.(DataType))
			}
		}
		for _, input := range node.Inputs {
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
