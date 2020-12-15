#version 330

layout (location = 0) in vec3 Position;                                             
layout (location = 1) in vec2 TexCoord;                                             
layout (location = 2) in ivec4 BoneIDs_1;
layout (location = 3) in vec4 Weights_1;
layout (location = 4) in ivec4 BoneIDs_2;
layout (location = 5) in vec4 Weights_2;
layout (location = 6) in ivec4 BoneIDs_3;
layout (location = 7) in vec4 Weights_3;


out vec2 out_tex_coord;
out vec4 out_vert_pos;

out float out_jointLabel;


uniform mat4 bone_matrix[26];
uniform mat4 model_view_proj_mat;
uniform mat4 model_view_mat;


void main(void)
{   
	mat4 boneTransform;
	
	boneTransform = bone_matrix[BoneIDs_1[0]] * Weights_1[0];
	for (int i = 1; i < 4; i++)
		boneTransform += bone_matrix[BoneIDs_1[i]] * Weights_1[i];
	for (int i = 0; i < 4; i++)
		boneTransform += bone_matrix[BoneIDs_2[i]] * Weights_2[i];
	for (int i = 0; i < 4; i++)
		boneTransform += bone_matrix[BoneIDs_3[i]] * Weights_3[i];


//modified	

	gl_Position = transpose(model_view_proj_mat) * boneTransform * vec4(Position.x,Position.y,Position.z,1.0);
	//gl_Position = model_view_proj_mat * boneTransform * vec4(Position.x,Position.y,Position.z,1.0);
	out_tex_coord = TexCoord;
	
	//add
	out_vert_pos=transpose(model_view_mat) * boneTransform * vec4(Position, 1.0);
	//out_vert_pos=gl_Position;


	//joint label
	/*
	float w[12];
	for(int i=0;i<12;i++){
		if(i<4)
			w[i]=Weights_1[i];
		else if(i>=4 && i<8)
			w[i]=Weights_2[i-4];
		else
			w[i]=Weights_3[i-8];
	}


	
	float w_min = 0;

	for(int i=0;i<12;i++){
		if(i<4){
			if(w[i]>w_min){
				w_min=w[i];
				out_jointLabel=BoneIDs_1[i];
			}
		}
		else if(i>=4 && i<8){
			if(w[i]>w_min){
				w_min=w[i];
				out_jointLabel=BoneIDs_2[i-4];		
			}
		}
		else{
			if(w[i]>w_min){
				w_min=w[i];
				out_jointLabel=BoneIDs_3[i-8];		
			}
		}
	}
	*/



//original
/*
	gl_Position = transpose(model_view_proj_mat) * boneTransform * vec4(Position, 1.0);	
	out_tex_coord = TexCoord;
	
	//add
	//out_vert_pos=transpose(model_view_mat) * boneTransform * vec4(Position, 1.0);
	out_vert_pos=gl_Position;
*/

	
}