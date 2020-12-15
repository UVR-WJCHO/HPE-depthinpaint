#version 330

layout(location = 0) in vec3 Position;

uniform mat4 model_view_proj_mat;
uniform mat4 model_view_mat;

out vec4 out_vert_pos;

void main(){
  gl_Position = transpose(model_view_proj_mat) * vec4(Position.x,Position.y,Position.z,1.0);

  out_vert_pos=transpose(model_view_mat) * vec4(Position, 1.0);
	
}