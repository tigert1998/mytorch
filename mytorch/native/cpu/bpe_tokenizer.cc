#include <fstream>
#include <iomanip>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using IndexType = uint32_t;
using MergeType = std::pair<IndexType, IndexType>;

struct BPETokenizerParams {
  std::vector<std::string> vocab;
  std::vector<MergeType> merges;
};

using FreqTableItem = uint64_t;

struct PQItem {
  MergeType merge;
  FreqTableItem item;
  bool operator<(const PQItem& other) const {
    return item < other.item || item == other.item && merge < other.merge;
  }
};

struct Node {
  IndexType index;
  Node *left, *right;
};

BPETokenizerParams TrainBPETokenizer(const std::string& raw_path,
                                     int num_merges, bool progress) {
  std::priority_queue<PQItem> pq;
  std::map<MergeType, FreqTableItem> freq_table;
  std::map<MergeType, std::set<Node*>> merge_to_nodes;
  std::vector<std::string> vocab;
  std::vector<MergeType> merges;
  vocab.resize(num_merges + 256);
  merges.reserve(num_merges);

  Node* head;
  {
    std::ifstream in_file(raw_path, std::ios::in | std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(in_file)),
                            (std::istreambuf_iterator<char>()));
    Node* nodes = new Node[bytes.size()];
    head = nodes;
    for (uint64_t i = 0; i < bytes.size(); i++) {
      nodes[i].index = uint32_t(uint8_t(bytes[i]));
      nodes[i].left = i == 0 ? nullptr : &nodes[i - 1];
      nodes[i].right = i + 1 < bytes.size() ? &nodes[i + 1] : nullptr;
    }
    in_file.close();
    for (int i = 0; i <= 255; i++) vocab[i] = std::string(1, i);
    for (uint64_t i = 0; i + 1 < bytes.size(); i++) {
      MergeType key = {nodes[i].index, nodes[i + 1].index};
      freq_table[key] += 1;
      merge_to_nodes[key].insert(&nodes[i]);
    }
    for (auto pair : freq_table) {
      pq.push({pair.first, pair.second});
    }
  }

  for (int merge_id = 0; merge_id < num_merges; merge_id++) {
    PQItem top;
    while (1) {
      top = pq.top();
      pq.pop();
      if (freq_table.count(top.merge) == 0) continue;
      if (!(freq_table[top.merge] == top.item)) {
        top.item = freq_table[top.merge];
        pq.push(top);
        continue;
      }
      break;
    }
    merges.push_back(top.merge);
    IndexType new_index = 256 + merge_id;
    vocab[new_index] = vocab[top.merge.first] + vocab[top.merge.second];

    std::set<MergeType> new_merges;
    for (Node* node : merge_to_nodes[top.merge]) {
      if (node->left != nullptr) {
        MergeType merge = {node->left->index, node->index};
        freq_table[merge] -= 1;
        merge_to_nodes[merge].erase(node->left);
        merge = {node->left->index, new_index};
        freq_table[merge] += 1;
        merge_to_nodes[merge].insert(node->left);
        new_merges.insert(merge);
      }
      Node* right_node = node->right;
      if (right_node->right != nullptr) {
        MergeType merge = {right_node->index, right_node->right->index};
        freq_table[merge] -= 1;
        merge_to_nodes[merge].erase(right_node);
        merge = {new_index, right_node->right->index};
        freq_table[merge] += 1;
        merge_to_nodes[merge].insert(node);
        new_merges.insert(merge);
      }
      node->index = new_index;
      node->right = right_node->right;
      if (right_node->right != nullptr) right_node->right->left = node;
    }

    merge_to_nodes.erase(top.merge);
    for (MergeType merge : new_merges) {
      pq.push({merge, freq_table[merge]});
    }

    if (progress) printf("Progress: %d/%d\r", merge_id + 1, num_merges);
  }

  delete[] head;

  return {vocab, merges};
}

std::string Escape(const std::string& vocab) {
  std::stringstream tmp;
  for (int j = 0; j < vocab.size(); j++) {
    const char g_caron_utf8[] = "\xC4\xA0";
    char c = vocab[j];
    if (c == '"')
      tmp << "\\\"";
    else if (c == ' ')
      tmp << g_caron_utf8;
    else if (c == '\\')
      tmp << "\\\\";
    else if (32 <= c && c <= 126)
      tmp << c;
    else if (c == 10)
      tmp << "\\n";
    else if (c == 13)
      tmp << "\\r";
    else
      tmp << "\\u" << std::setfill('0') << std::setw(4) << std::hex
          << uint32_t(uint8_t(c));
  }
  return tmp.str();
};

void ExportBPETokenizerParams(const BPETokenizerParams& params,
                              const std::string& json_path) {
  std::ofstream file(json_path,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  std::stringstream ss;
  ss << "{\"model\":{\"vocab\":{";
  for (int i = 0; i < params.vocab.size(); i++) {
    ss << "\"" + Escape(params.vocab[i]) + "\":" + std::to_string(i);
    if (i + 1 < params.vocab.size()) ss << ",";
  }
  ss << "},\"merges\":[";
  for (int i = 0; i < params.merges.size(); i++) {
    ss << "\"" + Escape(params.vocab[params.merges[i].first]) + " " +
              Escape(params.vocab[params.merges[i].second]) + "\"";
    if (i + 1 < params.merges.size()) ss << ",";
  }
  ss << "]}}";

  std::string data = ss.str();
  file.write(data.data(), data.size());
}

int main() {
  auto params = TrainBPETokenizer(
      "C:/Users/tiger/Projects/datasets/TinyStoriesV2-GPT4-valid.txt", 10000,
      true);
  ExportBPETokenizerParams(params, "./test.json");
}