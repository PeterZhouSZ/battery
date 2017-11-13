#include "RenderList.h"

void RenderList::clear() { _shaderToQueue.clear(); }

void RenderList::add(const std::shared_ptr<Shader> &shaderPtr,
                     RenderItem item) {
  _shaderToQueue[shaderPtr].push_back(item);
}

void RenderList::render() {

  for (auto it : _shaderToQueue) {
    auto &shader = *it.first;
    const auto &queue = it.second;

    shader.bind();

    for (auto &item : queue) {

      for (auto &shaderOpt : item.shaderOptions) {

        std::visit(
            [&](auto arg) {
              auto &res = shader[shaderOpt.first];
              res = arg;
            },
            shaderOpt.second);
      }

      item.vbo.render();
    }

    shader.unbind();
  }
}
