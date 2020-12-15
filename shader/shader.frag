/*
#version 330

in vec2 out_tex_coord;
in vec4 out_vert_pos;

uniform sampler2D textureObject;   

uniform int vis_mode;                                                             

void main(void)
{
	if(vis_mode==0) //color
		gl_FragColor = texture2D(textureObject, out_tex_coord);
		//gl_FragColor = vec4(1,0,0,1);
	else //depth
		gl_FragColor=vec4(out_vert_pos.x,out_vert_pos.y,out_vert_pos.z,1.0);
}
*/

#version 330

in vec2 out_tex_coord;
in vec4 out_vert_pos;

in float out_jointLabel;

uniform sampler2D textureObject;   
uniform int vis_mode;              


                                           

void main(void)
{

	if(vis_mode==0) //color 
	{
		if(out_vert_pos.z>500)
			gl_FragColor = vec4(0,0,0,0);
		else
			gl_FragColor = texture2D(textureObject, out_tex_coord);
	}
	else //depth
	{
		if(out_vert_pos.z>500)
			gl_FragColor=vec4(0,0,0,0);	
		else
			gl_FragColor=vec4(out_vert_pos.x,out_vert_pos.y,out_vert_pos.z,1.0);
	}

	//gl_FragColor=vec4(out_vert_pos.x,out_vert_pos.y,out_vert_pos.z,out_jointLabel);

	/*
	if(vis_mode==0) //color
	{
		gl_FragColor = texture2D(textureObject, out_tex_coord);
	}
	else //depth
	{
		gl_FragColor=vec4(out_vert_pos.x,out_vert_pos.y,out_vert_pos.z,1.0);
	}

	//gl_FragColor=vec4(out_vert_pos.x,out_vert_pos.y,out_vert_pos.z,out_jointLabel);
	*/

}








