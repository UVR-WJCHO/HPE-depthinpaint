#version 330

out vec4 color;
in vec4 out_vert_pos;
uniform int vis_mode; 
uniform int label;

void main(){

	if(vis_mode==0){
		if(label==1)
			color = vec4(1,0,0,1); //red
		else if (label==2)
			color = vec4(0,0,1,2); //blue
		else if (label==3)
			color = vec4(0,1,0,3); //green
	}
	else{
		color = vec4(out_vert_pos.x,out_vert_pos.y,out_vert_pos.z,label);
	}
}