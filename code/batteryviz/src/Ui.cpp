#include "Ui.h"

#include "BatteryApp.h"

#include "imgui.h"
#include "imgui/imgui_impl_glfw_gl3.h"

#include <glm/gtc/type_ptr.hpp>

#include <fstream>

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
		int ID = reinterpret_cast<int>(M);
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
			hasChanged |= renderOptionSet(it.first, it.second, depth + 1);
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


	int w = _app._window.width * 0.2f;
	ImGui::SetNextWindowPos(ImVec2(_app._window.width - w, 0), ImGuiSetCond_Always);
	ImGui::SetNextWindowSize(ImVec2(w, 2.0f * (_app._window.height / 3.0f)), ImGuiSetCond_Always);

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

	ImGui::SliderFloat3("Slice (Min)", reinterpret_cast<float*>(&_app._options["Render"].get<vec3>("sliceMin")), -1, 1);
	ImGui::SliderFloat3("Slice (Max)", reinterpret_cast<float*>(&_app._options["Render"].get<vec3>("sliceMax")), -1, 1);


	{
		_app._volumeRaycaster->sliceMin = _app._options["Render"].get<vec3>("sliceMin");
		_app._volumeRaycaster->sliceMax = _app._options["Render"].get<vec3>("sliceMax");

		_app._volumeRaycaster->opacityWhite = _app._options["Render"].get<float>("opacityWhite");
		_app._volumeRaycaster->opacityBlack = _app._options["Render"].get<float>("opacityBlack");
	}	


	ImGui::SliderFloat3("Quadric", reinterpret_cast<float*>(&_app._quadric), 0, 10);

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
