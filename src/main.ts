import DrawTraining from "./sketches/DrawTraining";
import NeuralNet from "./lib/NeuralNet";

import "./style.css";

const m = Math.random() * 2 - 1;

const n = Math.random() * 400;

const classify = (x: number, y: number) => {
  return m * x + n > y ? [1, 0] : [0, 1];
};

// const classify = (x: number, y: number) => {
//   return x * x + y * y > 200 * 200 ? [1, 0] : [0, 1];
// };

const dataset: iPoint[] = [];

for (let d = 0; d < 500; d++) {
  const x = Math.random() * 400;
  const y = Math.random() * 400;
  const label = classify(x, y);

  dataset.push({ x, y, label });
}

// active(x: number) {
//   return (2 * Math.atan(x)) / Math.PI;
// }

// pActive(x: number) {
//   return 2 / (Math.PI * (1 + x * x));
// }

const sigma = (x: number) => 1 / (1 + Math.exp(-x));

const sigmaPrime = (x: number) => sigma(x) * (1 - sigma(x));

const nn = new NeuralNet(2, 2);

nn.functions(sigma, sigmaPrime);

new DrawTraining(400, 400, dataset, nn);
