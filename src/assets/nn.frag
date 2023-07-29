precision highp float;

#define PI 3.1415926538

uniform vec2 resolution;

uniform int sizes[100];
uniform float weights[100];
uniform float biases[100];

float sigma(float x) {
  return 1.0 / (1.0 + exp(-x));
}

void main() {
  vec2 st = gl_FragCoord.xy / (2.0 * resolution);

  float a0 = sigma(weights[0] * st.x + weights[1] * st.y + biases[0]);
  float a1 = sigma(weights[2] * st.x + weights[3] * st.y + biases[1]);
  float a2 = sigma(weights[4] * st.x + weights[5] * st.y + biases[2]);

  float b0 = sigma(a0 * weights[6] + a1 * weights[7] + a2 * weights[8] + biases[3]);
  float b1 = sigma(a0 * weights[9] + a1 * weights[10] + a2 * weights[11] + biases[4]);
  float b2 = sigma(a0 * weights[12] + a1 * weights[13] + a2 * weights[14] + biases[5]);

  gl_FragColor = vec4(b0, b1, b2, 1.0);
}