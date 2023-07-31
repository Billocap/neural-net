/// <reference types="vite/client" />

/**
 * One dimensional array of numbers.
 */
type Vector = number[];
/**
 * Bidimensional array of numbers.
 */
type Matrix = number[][];

/**
 * Namespace containing the basic types for a neural network.
 */
namespace Model {
  /**
   * Defines an activation function or the derivative of an activation function.
   */
  type Activation = (x: number) => number;

  /**
   * Defines the gradient for a single layer.
   *
   * The first value is the gradient for the weights and the second the gradient for the biases.
   */
  type Gradient = [Matrix, Vector];

  /**
   * Represents a single layer.
   */
  interface Layer {
    /**
     * Size of the layer in the form `[width, height]`.
     */
    size: [number, number];
    /**
     * Matrix of size `[width, height]` containing the layer's weights.
     */
    weights: Matrix;
    /**
     * Vector of size `[height]` containing the layer's biases.
     */
    biases: Vector;
  }

  /**
   * Represents a full neural net.
   */
  interface Model {
    /**
     * Array containing the layers for neural net.
     */
    layers: Layer[];
  }
}

/**
 * Namespace for handling data.
 */
namespace Data {
  /**
   * Function that generates value vectors.
   */
  type Generator = () => Vector;
  /**
   * A classifier returns a label vector for the given value vector.
   */
  type Classifier = (data: Vector) => Vector;

  /**
   * A single data receives a value and a label (expected output).
   */
  interface Data {
    /**
     * Actual value for the data.
     */
    value: Vector;
    /**
     * Expected output.
     */
    label: Vector;
  }

  /**
   * Collection of data.
   */
  type Set = Array<Data>;
}
