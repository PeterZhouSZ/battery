#include "Ui.h"

#include "BatteryApp.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw_gl3.h"

#include "imgui/imgui_file_explorer.h"

#include <glm/gtc/type_ptr.hpp>

#include <batterylib/include/VolumeIO.h>
#include <batterylib/include/VolumeMeasures.h>
#include <batterylib/include/VolumeGenerator.h>
#include <batterylib/include/VolumeSegmentation.h>
#include <batterylib/include/VolumeSurface.h>


#include <chrono>
#include <iostream>
#include <fstream>
#include <batterylib/include/Timer.h>

void mayaStyle() {
	ImGuiStyle& style = ImGui::GetStyle();

	style.ChildWindowRounding = 3.f;
	style.GrabRounding = 0.f;
	style.WindowRounding = 0.f;
	style.ScrollbarRounding = 3.f;
	style.FrameRounding = 3.f;
	style.WindowTitleAlign = ImVec2(0.5f, 0.5f);

	style.Colors[ImGuiCol_Text] = ImVec4(0.73f, 0.73f, 0.73f, 1.00f);
	style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
	style.Colors[ImGuiCol_WindowBg] = ImVec4(0.26f, 0.26f, 0.26f, 0.95f);
	style.Colors[ImGuiCol_ChildWindowBg] = ImVec4(0.28f, 0.28f, 0.28f, 1.00f);
	style.Colors[ImGuiCol_PopupBg] = ImVec4(0.26f, 0.26f, 0.26f, 1.00f);
	style.Colors[ImGuiCol_Border] = ImVec4(0.26f, 0.26f, 0.26f, 1.00f);
	style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.26f, 0.26f, 0.26f, 1.00f);
	style.Colors[ImGuiCol_FrameBg] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
	style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
	style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
	style.Colors[ImGuiCol_TitleBg] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.26f, 0.26f, 0.26f, 1.00f);
	style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.21f, 0.21f, 0.21f, 1.00f);
	style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_ComboBg] = ImVec4(0.32f, 0.32f, 0.32f, 1.00f);
	style.Colors[ImGuiCol_CheckMark] = ImVec4(0.78f, 0.78f, 0.78f, 1.00f);
	style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.74f, 0.74f, 0.74f, 1.00f);
	style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.74f, 0.74f, 0.74f, 1.00f);
	style.Colors[ImGuiCol_Button] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.43f, 0.43f, 0.43f, 1.00f);
	style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.11f, 0.11f, 0.11f, 1.00f);
	style.Colors[ImGuiCol_Header] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_Column] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
	style.Colors[ImGuiCol_ColumnHovered] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	style.Colors[ImGuiCol_ColumnActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	style.Colors[ImGuiCol_ResizeGrip] = ImVec4(0.36f, 0.36f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	style.Colors[ImGuiCol_CloseButton] = ImVec4(0.59f, 0.59f, 0.59f, 1.00f);
	style.Colors[ImGuiCol_CloseButtonHovered] = ImVec4(0.98f, 0.39f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_CloseButtonActive] = ImVec4(0.98f, 0.39f, 0.36f, 1.00f);
	style.Colors[ImGuiCol_PlotLines] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
	style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
	style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
	style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
	style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.32f, 0.52f, 0.65f, 1.00f);
	style.Colors[ImGuiCol_ModalWindowDarkening] = ImVec4(0.20f, 0.20f, 0.20f, 0.50f);
}


struct ImguiInputs {



	template <typename T>
	bool operator() (T & value) {
		ImGui::Text("Type Not implemented");
		return false;
	}

	bool operator() (float & value) {
		return ImGui::InputFloat("##value", &value, value / 100.0f);
	}

	bool operator() (int & value) {
		return ImGui::InputInt("##value", &value, value / 100);
	}

	bool operator() (bool & value) {
		bool res = ImGui::Checkbox("##value", &value);

		if (res) {
			char br;
			br = 0;
		}
		return res;
	}

	bool operator() (double & value) {
		float fv = static_cast<float>(value);
		bool changed = (*this)(fv);
		if (changed) {
			value = static_cast<double>(fv);
			return true;
		}
		return false;
	}

	bool operator() (std::string & value) {
		value.reserve(512);
		if (value.length() > 512)
			value.resize(512);
		return ImGui::InputTextMultiline("##value", &value[0], 512);
	}

	bool operator() (vec2 & value) {
		return ImGui::DragFloat2("##value", glm::value_ptr(value), 0.1f);
	}
	bool operator() (vec3 & value) {
		return ImGui::DragFloat3("##value", glm::value_ptr(value), 0.1f);
	}
	bool operator() (vec4 & value) {
		return ImGui::DragFloat4("##value", glm::value_ptr(value), 0.1f);
	}

	bool operator() (ivec2 & value) {
		return ImGui::DragInt2("##value", glm::value_ptr(value), 1);
	}
	bool operator() (ivec3 & value) {
		return ImGui::DragInt3("##value", glm::value_ptr(value), 1);
	}
	bool operator() (ivec4 & value) {
		return ImGui::DragInt4("##value", glm::value_ptr(value), 1);
	}

	bool operator() (mat2 & value) {
		return renderMatrix(glm::value_ptr(value), 2);
	}

	bool operator() (mat3 & value) {
		return renderMatrix(glm::value_ptr(value), 3);
	}

	bool operator() (mat4 & value) {
		return renderMatrix(glm::value_ptr(value), 4);
	}

private:

	bool renderMatrix(float * M, int dimension) {
		int ID = static_cast<int>(reinterpret_cast<size_t>(M));
		ImGui::BeginChildFrame(ID, ImVec2(ImGui::GetColumnWidth() - 17, ImGui::GetItemsLineHeightWithSpacing() * dimension + 5));
		ImGui::Columns(dimension);

		bool changed = false;
		for (int k = 0; k < dimension; k++) {
			for (int i = 0; i < dimension; i++) {
				ImGui::PushID(k*dimension + i);
				changed |= ImGui::DragFloat("##value", M + k + i*dimension);
				ImGui::PopID();
			}
			if (k < dimension - 1)
				ImGui::NextColumn();
		}

		ImGui::Columns(1);
		ImGui::EndChildFrame();
		return changed;
	}
};

bool renderOptionSet(const std::string & name, OptionSet & options, unsigned int depth)
{

	bool hasChanged = false;

	if (depth == 0) {
		ImGui::Columns(2);
		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
	}

	const void * id = &options;
	ImGui::PushID(id);
	ImGui::AlignFirstTextHeightToWidgets();

	bool isOpen = (depth > 0) ? ImGui::TreeNode(id, "%s", name.c_str()) : true;


	ImGui::NextColumn();
	ImGui::AlignFirstTextHeightToWidgets();

	ImGui::NextColumn();

	if (isOpen) {

		for (auto & it : options.children) {
			hasChanged |= renderOptionSet(it.first, *it.second, depth + 1);
		}


		for (auto & opt : options.options) {
			const void * oid = &opt;
			ImGui::PushID(oid);

			ImGui::Bullet();
			ImGui::Selectable(opt.first.c_str());

			ImGui::NextColumn();
			ImGui::PushItemWidth(-1);
			ImGui::AlignFirstTextHeightToWidgets();

			hasChanged |= std::visit(ImguiInputs(), opt.second.value);

			ImGui::PopItemWidth();
			ImGui::NextColumn();



			ImGui::PopID();
		}


		if (depth > 0)
			ImGui::TreePop();

	}

	ImGui::PopID();


	if (depth == 0) {
		ImGui::Columns(1);
		ImGui::PopStyleVar();
	}

	return hasChanged;

}


Ui::Ui(BatteryApp & app) : _app(app)
{
	//gui
	ImGui_ImplGlfwGL3_Init(_app._window.handle, false);
	//mayaStyle();
}

void Ui::update(double dt)
{
	ImGui_ImplGlfwGL3_NewFrame();


	int w = static_cast<int>(_app._window.width * 0.2f);
	ImGui::SetNextWindowPos(
		ImVec2(static_cast<float>(_app._window.width - w), 0), 
		ImGuiSetCond_Always
	);

	if (_app._options["Render"].get<bool>("slices")) {
		ImGui::SetNextWindowSize(
			ImVec2(static_cast<float>(w), 2.0f * (static_cast<float>(_app._window.height) / 3.0f)),
			ImGuiSetCond_Always
		);
	}
	else {
		ImGui::SetNextWindowSize(
			ImVec2(static_cast<float>(w), static_cast<float>(_app._window.height)),
			ImGuiSetCond_Always
		);
	}

	static bool mainOpen = false;

	
	

	ImGui::Begin("Main", &mainOpen);

	//FPS display
	{
		double fps = _app.getFPS();
		ImVec4 color;
		if (fps < 30.0)
			color = ImVec4(1, 0, 0,1);
		else if (fps < 60.0)
			color = ImVec4(0, 0.5, 1,1);
		else
			color = ImVec4(0, 1, 0, 1);

		ImGui::TextColored(color, "FPS: %f", float(fps));
	}


	/*
		Options
	*/
	ImGui::Separator();
	ImGui::Text("Options");
	ImGui::Separator();

	if (ImGui::Button("Save")) {
		std::ofstream optFile(OPTIONS_FILENAME);		
			optFile << _app._options;		
	}
	ImGui::SameLine();
	if (ImGui::Button("Load")) {
		std::ifstream optFile(OPTIONS_FILENAME);
		optFile >> _app._options;
	}

	renderOptionSet("Options", _app._options, 0);

	
	ImGui::Separator();
	ImGui::Text("View");
	ImGui::Separator();

	ImGui::SliderFloat3("Slice (Min)", reinterpret_cast<float*>(&_app._options["Render"].get<vec3>("sliceMin")), -1, 1);
	ImGui::SliderFloat3("Slice (Max)", reinterpret_cast<float*>(&_app._options["Render"].get<vec3>("sliceMax")), -1, 1);


	{
		_app._volumeRaycaster->sliceMin = _app._options["Render"].get<vec3>("sliceMin");
		_app._volumeRaycaster->sliceMax = _app._options["Render"].get<vec3>("sliceMax");

		_app._volumeRaycaster->opacityWhite = _app._options["Render"].get<float>("opacityWhite");
		_app._volumeRaycaster->opacityBlack = _app._options["Render"].get<float>("opacityBlack");

		_app._volumeRaycaster->preserveAspectRatio = _app._options["Render"].get<bool>("preserveAspectRatio");

		_app._volumeRaycaster->showGradient = _app._options["Render"].get<bool>("showGradient");


		_app._volumeRaycaster->setNormalizeRange(
		{ 
			_app._options["Render"].get<float>("normalizeLow"), 
			_app._options["Render"].get<float>("normalizeHigh")
		}
		);		
	}	

	

	

	ImGui::SliderInt("RenderChannel", 
		&_app._options["Render"].get<int>("channel"),
		0, _app._volume->numChannels() - 1
	);

	
	if (_app._volume->numChannels() > 0) {
		int channelID = _app._options["Render"].get<int>("channel");
		if (channelID >= _app._volume->numChannels()) {
			channelID = _app._volume->numChannels() - 1;
			_app._options["Render"].set<int>("channel", channelID);
		}
		auto & renderChannel = _app._volume->getChannel(channelID);

		ImGui::TextColored(ImVec4(1, 1, 0, 1), "%s",
			renderChannel.getName().c_str()
		);
		ImGui::TextColored(ImVec4(1, 1, 0, 1), "%d %d %d",
			renderChannel.dim().x, renderChannel.dim().y, renderChannel.dim().z
		);

		ImGui::SameLine(); 

		if(ImGui::Button("Clear Channel"))
			renderChannel.clear();

		ImGui::SameLine();
		if (ImGui::Button("Normalize range")) {
			
			uchar buf[64];
			renderChannel.min(buf);
			float minVal = primitiveToNormFloat(renderChannel.type(), buf);
			renderChannel.max(buf);
			float maxVal = primitiveToNormFloat(renderChannel.type(), buf);

			std::cout << "Min: " << minVal << ", Max: " << maxVal << std::endl;
						
			_app._options["Render"].set<float>("normalizeLow", float(minVal));
			_app._options["Render"].set<float>("normalizeHigh",float(maxVal));

		}

	}
	else {
		ImGui::TextColored(ImVec4(1, 0, 0, 1), "No VolumeChannels Loaded");
	}


	/*
	Volume
	*/

	
	
	
	ImGui::Separator();
	ImGui::Text("Measurements");
	ImGui::Separator();

	static bool enableMGGPU = false;
	static bool enableBiCGCPU = false;
	static bool enableBiCGGPU = true;	

	
	if (ImGui::Button("Tortuosity")) {
		

		/*
		Prepare
		*/
		blib::TortuosityParams tp;
		tp.coeffs = {
			_app._options["Diffusion"].get<float>("D_zero"),
			_app._options["Diffusion"].get<float>("D_one")
		};
		tp.dir = Dir(_app._options["Diffusion"].get<int>("direction"));
		tp.tolerance = pow(10.0, -_app._options["Diffusion"].get<int>("Tolerance"));
		tp.maxIter = size_t(_app._options["Diffusion"].get<int>("maxIter"));

		auto & mask = _app._volume->getChannel(CHANNEL_MASK);					

		blib::Timer tt(true);

		if (enableMGGPU) {
			auto tau = blib::getTortuosity<double>(mask, tp, blib::DiffusionSolverType::DSOLVER_MGGPU, &_app._volume->getChannel(CHANNEL_CONCETRATION));
			std::cout << "MGGPU\t\t" << tau << std::endl;
		}

		if (enableBiCGGPU) {
			auto tau = blib::getTortuosity<double>(mask, tp, blib::DiffusionSolverType::DSOLVER_BICGSTABGPU, &_app._volume->getChannel(CHANNEL_CONCETRATION));
			std::cout << "BICGSTABGPU\t\t" << tau << std::endl;
		}

		if (enableBiCGCPU) {
			mask.getCurrentPtr().allocCPU();
			mask.getCurrentPtr().retrieve();

			auto tau = blib::getTortuosity<double>(mask, tp, blib::DiffusionSolverType::DSOLVER_EIGEN, &_app._volume->getChannel(CHANNEL_CONCETRATION));
			std::cout << "BiCGCPU\t\t" << tau << std::endl;
		}

		std::cout << "Tau time: " << tt.time() << std::endl;
	}

	ImGui::SameLine();
	ImGui::Checkbox("MG", &enableMGGPU); ImGui::SameLine();
	ImGui::Checkbox("BiCG-C", &enableBiCGCPU); ImGui::SameLine();
	ImGui::Checkbox("BiCG-G", &enableBiCGGPU); 

	
	static bool enableRADMesh = true;
	
	
	if (ImGui::Button("Reactive Area Density")) {
		auto ccl = blib::getVolumeCCL(_app._volume->getChannel(CHANNEL_MASK), 255);


		if (enableRADMesh) {
			Dir dir = Dir(_app._options["Diffusion"].get<int>("direction"));
			blib::VolumeChannel boundaryVolume = blib::generateBoundaryConnectedVolume(ccl, dir);
			boundaryVolume.getCurrentPtr().createTexture();

			//blib::getVolumeArea(boundaryVolume);			
			//_app._volume->emplaceChannel(std::move(boundaryVolume));
			//_app._volume->emplaceChannel(std::move(areas));

			uint vboIndex;
			size_t Nverts = 0;
			getVolumeAreaMesh(boundaryVolume, &vboIndex, &Nverts);
			if (Nverts > 0) {
				_app._volumeMC = std::move(VertexBuffer<VertexData>(vboIndex, Nverts));
			}
			else {
				_app._volumeMC = std::move(VertexBuffer<VertexData>());
			}

		}

		auto rad = blib::getReactiveAreaDensityTensor<double>(ccl);

		for (auto i = 0; i < 6; i++) {
			std::cout << "RAD (dir: " << i << "): " << rad[i] << std::endl;
		}

	}
	ImGui::SameLine();
	ImGui::Checkbox("Gen. Mesh", &enableRADMesh);
	



	static bool concGraph = false;
	ImGui::Checkbox("Concetration Graph", &concGraph);
	
	if(concGraph){

		ImGui::SameLine();

		const uint channel = CHANNEL_CONCETRATION;
		const Dir dir = Dir(_app._options["Diffusion"].get<int>("direction"));		
		
		static std::vector<double> vals;	


		auto & c = _app._volume->getChannel(channel);
		assert(c.type() == TYPE_DOUBLE);
		
		if (ImGui::Button("Refresh")) {
			vals.resize(c.dimInDirection(dir), 0.0f);
			c.sumInDir(dir, vals.data());

			auto sliceElemCount = float(c.sliceElemCount(dir));
			for (auto & v : vals)
				v /= sliceElemCount;

		}

		{
			std::vector<float> tmp;
			for (auto f : vals) tmp.push_back(float(f));

			ImGui::PlotLines("C", tmp.data(), int(tmp.size()), 0, nullptr, 0.0f, 1.0f, ImVec2(200, 300));
		}	
	}

	ImGui::Separator();
	ImGui::Text("Load .TIFF");
	ImGui::Separator();
	{
		static std::string curDir = "../../data";
		std::string filename;
		std::tie(curDir, filename) = imguiFileExplorer(curDir, ".tiff", true);
		if (filename != "") {
			_app.loadFromFile(curDir);
		}
	}

	{
		if (ImGui::Button("Load Default")) {
			_app.loadFromFile(_app._options["Input"].get<std::string>("DefaultPath"));
		}

		ImGui::SameLine();

		if (ImGui::Button("Reset")) {
			_app.reset();
		}

	}

	ImGui::Separator();
	ImGui::Text("Generate volume");
	ImGui::Separator();

	


	renderOptionSet("Generator Options", _app._options["Generator"], 0);

	int genResX = _app._options["Generator"].get<int>("Resolution");
	ivec3 genRes = ivec3(genResX);

	if (ImGui::Button("Spheres")) {
		auto & opt = _app._options["Generator"]["Spheres"];

		blib::GeneratorSphereParams p;
		p.N = opt.get<int>("N");
		p.rmin = opt.get<float>("RadiusMin");
		p.rmax = opt.get<float>("RadiusMax");
		p.maxTries = opt.get<int>("MaxTries");
		p.overlapping = opt.get<bool>("Overlapping");
		p.withinBounds = opt.get<bool>("WithinBounds");


		
		bool result;
		_app.loadFromMask(
			blib::rasterizeSpheres(genRes, blib::generateSpheres(p))
		);

		if (!result) {
			std::cerr << "Failed to generate spheres" << std::endl;
		}
	}

	ImGui::SameLine();
	if (ImGui::Button("Max Spheres")) {
		auto & opt = _app._options["Generator"]["Spheres"];

		blib::GeneratorSphereParams p;
		p.N = opt.get<int>("N");
		p.maxTries = opt.get<int>("MaxTries");
		p.rmin = opt.get<float>("RadiusMin");
		p.rmax = opt.get<float>("RadiusMax");		
		p.withinBounds = opt.get<bool>("WithinBounds");
		p.overlapping = false;

		
		while (true) {

			bool result;
			_app.loadFromMask(
				blib::rasterizeSpheres(genRes, blib::generateSpheres(p))
			);

			if (!result) break;
			p.N += 1;
		}

		
		std::cerr << "Max N " << p.N << std::endl;
		
	}

	if (ImGui::Button("Filled")) {
		auto & opt = _app._options["Generator"]["Filled"];		
		_app.loadFromMask(
			blib::generateFilledVolume(genRes, uchar(opt.get<int>("value")))
		);
	}


	ImGui::Separator();

	static int backgroundValue = 0;
	ImGui::InputInt("Background CCL", &backgroundValue, 8, 255);


	if (ImGui::Button("CCL")) {

		
		auto ccl = getVolumeCCL(_app._volume->getChannel(0), backgroundValue);
		
		for (auto i = 0; i < 6; i++) {
			auto vol = generateBoundaryConnectedVolume(ccl, Dir(i));			
			_app._volume->emplaceChannel(generateCCLVisualization(ccl, &vol));
			//_app._volume->emplaceChannel(generateBoundaryConnectedVolume(ccl, Dir(i)));
		}

		_app._volume->emplaceChannel(generateCCLVisualization(ccl));
		

	}

	ImGui::End();


	ImGui::Render();
}

void Ui::callbackMouseButton(GLFWwindow * w, int button, int action, int mods)
{
	ImGui_ImplGlfwGL3_MouseButtonCallback(w, button, action, mods);
}

void Ui::callbackKey(GLFWwindow * w, int key, int scancode, int action, int mods)
{
	
	ImGui_ImplGlfwGL3_KeyCallback(w, key, scancode, action, mods);
}

void Ui::callbackScroll(GLFWwindow * w, double xoffset, double yoffset)
{
	ImGui_ImplGlfwGL3_ScrollCallback(w, xoffset, yoffset);
}

void Ui::callbackChar(GLFWwindow * w, unsigned int code)
{
	
	ImGui_ImplGlfwGL3_CharCallback(w, code);
}

bool Ui::isFocused() const
{
	return ImGui::IsAnyItemActive();
}

