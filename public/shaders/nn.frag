#ifdef GL_ES
precision mediump float;
#endif

#define PI 3.1415926538

uniform vec2 resolution;
uniform vec3 values;

float sigma(float x) {
  return 2.0 * atan(x) / PI;
}

void main() {
  vec2 st = gl_FragCoord.xy / resolution;

  float r = st.x * values.x + st.y * values.y + values.z;

  gl_FragColor = vec4(sigma(r) > 0.0 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0), 1.0);
}