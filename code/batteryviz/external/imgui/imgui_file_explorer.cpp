#include "imgui_file_explorer.h"
#include "imgui.h"

#if defined(__GNUC__)
    #include <experimental/filesystem>
#else
	#include <filesystem>
#endif

#include <sstream>

using namespace std;
namespace fs = std::experimental::filesystem;


std::string filesizeString(uintmax_t size) {

	std::stringstream ss;

	if (size < 1024)
		ss << size << " B";
	else if(size < 1024*1024)
		ss << size/1024 << " kB";
	else if (size < 1024 * 1024 * 1024)
		ss << size / (1024*1024) << " MB";
	else if (size < uintmax_t(1024) * 1024 * 1024 * 1024)
		ss << size / (1024 * 1024 * 1024) << " GB";

	return ss.str();
}

tuple<string, string>  imguiFileExplorer(
	const string & directory, 
	const string & extension, 
	bool canDelete /*= false */)
{

	
	std::hash<std::string> hash_fn;
	int ID = static_cast<int>(hash_fn(directory));
	fs::path path((directory.length() == 0) ? fs::current_path() : fs::path(directory));

	tuple<string, string> result = { path.string(), "" };

	

	//std::string dirPath = dirPathIn;

	ImGui::PushID(ID);

	ImGui::BeginChildFrame(ID, ImVec2(ImGui::GetWindowContentRegionWidth(), 400));

	ImGui::Text(path.string().c_str());

	ImGui::Columns(3);	


	if (ImGui::Selectable("..", false)) {
		std::get<0>(result) = path.parent_path().string();
	}

	for (auto & f : fs::directory_iterator(path)) {

		const auto & p = f.path();

		bool isDir = fs::is_directory(f);

		if (!isDir && extension != "" && extension != ".*"){
			auto ext = p.extension().string();
			if (p.extension().string() != extension) continue;
		}
		

		if (ImGui::Selectable(p.filename().string().c_str(), false)) {
			if (isDir) {
				std::get<0>(result) = p.string();				
			}
			else {
				std::get<1>(result) = p.string();
			}
		}

		ImGui::NextColumn();
		if (isDir) {
			ImGui::Text("DIR");
		}
		else {			
			ImGui::Text(filesizeString(fs::file_size(f)).c_str());
		}
		//..

		ImGui::NextColumn();
		//..


		ImGui::NextColumn();	

	}

	ImGui::Columns(1);

	ImGui::EndChildFrame();
	ImGui::PopID();

	return result;


}
