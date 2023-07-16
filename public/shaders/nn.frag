#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform vec3 u_values;

float sigma(float x) {
  return 1.0 / (1.0 + exp(-x));
}

void main() {
  vec2 st = gl_FragCoord.xy / u_resolution.xy;

  float r = u_values.x * gl_FragCoord.x + u_values.y * gl_FragCoord.y + u_values.z;

  gl_FragColor = vec4(vec3(sigma(r)), 1.0);
}