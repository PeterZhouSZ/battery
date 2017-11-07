#include "GLRenderer.h"


void RenderList::clear()
{
	_shaderToQueue.clear();
}


void RenderList::add(const std::shared_ptr<Shader> & shaderPtr, RenderItem item)
{
	_shaderToQueue[shaderPtr].push_back(item);
}

void RenderList::render()
{

	for (auto it : _shaderToQueue) {		
		auto & shader = *it.first;
		const auto & queue = it.second;

		for (auto & item : queue) {

			for (auto & shaderOpt : item.shaderOptions) {

				std::visit([&](auto arg) { 
					auto &res = shader[shaderOpt.first];
					res = arg;
				}, shaderOpt.second);
			}

			item.vbo.render();
		}

	}
	
}

bool MeshObject::_updateBuffer() const
{	
	std::vector<VertexData> data;
	data.reserve(_mesh.size() * 3);	
	
	VertexData vd;	
	vd.color[0] = 1.0f;
	vd.color[1] = 1.0f;
	vd.color[2] = 1.0f;
	vd.color[3] = 1.0f;
	vd.uv[0] = 0.0f;
	vd.uv[1] = 0.0f;	

	for (auto t : _mesh) {
		const auto N = t.normal();
		memcpy(&vd.normal, N.data(), N.SizeAtCompileTime * sizeof(float));

		for (auto v : t.v) {			
			memcpy(&vd.pos, v.data(), v.SizeAtCompileTime * sizeof(float));			
			data.push_back(vd);
		}		
	}

	return _buffer.setData(
		data.begin(), data.end()
	);

}

ShaderType MeshObject::getShaderType() const
{
	return ShaderType::PHONG;
}

ShaderOptions MeshObject::getShaderOptions(const Camera & cam, mat4 parentTransform) const
{
	ShaderOptions opt;

	auto M = parentTransform * transform;
	auto NM = glm::inverse(glm::transpose(glm::mat3(cam.getView() * M)));

	opt.push_back({ "M", M });
	opt.push_back({ "NM", NM });
	opt.push_back({ "PVM", cam.getPV() * M });


	return opt;
}

void renderSceneObject(const std::unordered_map<ShaderType, std::shared_ptr<Shader>> & shaders, const SceneObject & root, const Camera & camera)
{
	RenderList rl;

	std::stack<const SceneObject * > objStack;
	std::stack<mat4> transformStack;
	objStack.push(&root);
	transformStack.push(mat4(1.0f)); //start with identity


	while (!objStack.empty()) {

		auto & obj = *objStack.top();
		objStack.pop();

		auto & currentTransform = transformStack.top();
		transformStack.pop();

		{
			RenderList::RenderItem item = { obj.getVBO(), obj.getShaderOptions(camera, currentTransform) };

			//todo check if exists
			rl.add(shaders.find(obj.getShaderType())->second, item);
			//item.shaderOptions.push_back();
		}

		for (auto & child : obj.children) {
			objStack.push(child.get());

		}
	}
}
