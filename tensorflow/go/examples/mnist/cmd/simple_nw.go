// Copyright © 2016 Berner Setterwall
//

package cmd

import (
    "encoding/json"
	"fmt"
    "io/ioutil"
    "log"
    "time"
    "strings"

    tf "github.com/berset/tensorflow/tensorflow/go"

	"github.com/spf13/cobra"
)

// linregCmd represents the linreg command
var simpleNWCmd = &cobra.Command{
	Use:   "simple_nw",
	Short: "A brief description of your command",
	Long: `A longer description that spans multiple lines and likely contains examples
and usage of using your command. For example:

Cobra is a CLI library for Go that empowers applications.
This application is a tool to generate the needed files
to quickly create a Cobra application.`,
	Run: func(cmd *cobra.Command, args []string) {
        simpleNW()
	},
}

var Model string
var NWWeights string
var SampleFile string
var Dropout float32

func init() {
	RootCmd.AddCommand(simpleNWCmd)
    simpleNWCmd.Flags().StringVarP(&Model, "model", "m", "../../sample_graphs/mnist.pbtxt", "MNIST model")
    simpleNWCmd.Flags().StringVarP(&NWWeights, "nw-weights", "w", "weights.json", "Network weights to load")
    simpleNWCmd.Flags().StringVarP(&SampleFile, "sample", "s", "/Users/berset/git/tl2/ex/4/3069.json", "Sample file with image tensor and classification")
    simpleNWCmd.Flags().Float32VarP(&Dropout, "dropout", "d", 1.0, "Dropout (keep) probability")
}

type MNISTSample struct {
    Y      float32   `json:"y"`
    Tensor []float32 `json:"tensor"`
}

func predictNW(s *tf.Session, ns map[string]tf.Output, file string) (*float32, *float32, error) {

    jsonBlob, err := ioutil.ReadFile(file)
    if err != nil {
        return nil, nil, err
    }
    var sample MNISTSample
    err = json.Unmarshal(jsonBlob, &sample)
    if err != nil {
        return nil, nil, err
    }

    t1, _ := tf.NewTensor(sample.Tensor)

    inputs := make(map[tf.Output]*tf.Tensor)
    inputs[ns["x"]] = t1

    mkMat := func(x float32, n int) [][]float32 {
        var mat [][]float32
        mat = append(mat, []float32{})
        for i := 0; i < n; i++ {
            mat[0] = append(mat[0], x)
        }
        return mat
    }
    ph1, _ := tf.NewTensor(mkMat(Dropout, 784))
    ph2, _ := tf.NewTensor(mkMat(Dropout, 800))
    ph3, _ := tf.NewTensor(mkMat(Dropout, 800))
    inputs[ns["Placeholder"]]   = ph1
    inputs[ns["Placeholder_1"]] = ph2
    inputs[ns["Placeholder_2"]] = ph3

    output_targets := []string{
        "Identity",
        }

    outputs_spec := []tf.Output{}
    for _, ot := range output_targets {
        outputs_spec = append(outputs_spec, ns[ot])
    }

    outputs, err := s.Run(inputs, outputs_spec, nil)

    if err == nil {
        preds := outputs[0].Value().([][]float32)
        f := float32(0)
        max := preds[0][0]
        for i, v := range preds[0] {
            if v > max {
                f = float32(i)
                max = v
            }
        }
        return &f, &sample.Y, nil
    } else {
        log.Println(err)
        return nil, nil, err
    }
}

func simpleNW() {
    log.Println("starting")
    start := time.Now()
    g, ns, err := tf.LoadGraph(models("mnist"))
    if err != nil {
        fmt.Println(err)
    }
    log.Printf("load NN took %s\n", time.Since(start))

    start = time.Now()

    s, err := tf.NewSession(g, &tf.SessionOptions{})

    setup(s, ns)

    log.Printf("setup NN took %s\n", time.Since(start))

    start = time.Now()

    initOps, err := tf.InitWeights(g, ns, NWWeights)
    if err != nil {
        fmt.Println(err)
    }
    _, err = s.Run(nil, nil, initOps)
    if err != nil {
        fmt.Println(err)
    }

    log.Printf("initialzie NN took %s\n", time.Since(start))

    start = time.Now()

    _y, y, err := predictNW(s, ns, SampleFile)
    if err != nil {
        log.Println(err)
    } else {
        log.Printf("predicted: %d vs real: %d\n", int(*_y), int(*y))
    }

    log.Printf("forward NN took %s\n", time.Since(start))
}

func models(name string) string {
    path := "/Users/berset/git/gopath/"+
            "src/github.com/berset/tensorflow/tensorflow/go/sample_graphs/"
    return fmt.Sprintf("%s/%s.pbtxt", path, name)
}

func setup(s *tf.Session, ns map[string]tf.Output) error {
    ops := []*tf.Operation{}
    for key, value := range ns {
        if strings.HasSuffix(key, "/Assign") {
            ops = append(ops, value.Op)
        }
    }

    _, err := s.Run(nil, nil, ops)

    return err
}

