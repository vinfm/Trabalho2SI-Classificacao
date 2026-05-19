#include "TreeBuilder.hpp"
#include "Criteria.hpp"
#include <numeric>

TreeBuilder::TreeBuilder(TreeType treeType, Dataset* trainData, Dataset* subtrainData, int min_samples_leaf, int max_depth, int min_samples_split, double min_impurity_split, std::size_t n_features)
    : treeType_(treeType),
      trainData_(trainData),
      subtrainData_(subtrainData),
      min_samples_leaf_(min_samples_leaf),
      max_depth_(max_depth),
      min_samples_split_(min_samples_split),
      min_impurity_split_(min_impurity_split),
      n_features_(n_features),
      impurity_measure_(treeType == TreeType::Classification ? CriterionType::Gini : CriterionType::MSE),
      splitter_(treeType == TreeType::Classification ? CriterionType::Gini : CriterionType::MSE),
      gen_(std::random_device()())
{
    max_features_ = n_features_;
}

DecisionTree* TreeBuilder::CARTBuild()
{
    splitter_ = Splitter(
        impurity_measure_,
        subtrainData_->X,
        subtrainData_->Y,
        subtrainData_->ninputs(),
        max_features_,
        min_samples_leaf_);
    
    splitter_.setRandomGenerator(&gen_);

    // create root node and initialize sample indices
    Node root;
    root.isLeaf = false;
    root.depth = 0;
    root.sampleIndices.resize(subtrainData_->nrows());
    std::iota(root.sampleIndices.begin(), root.sampleIndices.end(), 0);

    // create tree with root node (DecisionTree constructor will push root)
    CART* tree = new CART(root, treeType_);
    std::vector<Node>& nodes = tree->nodes;

    // compute root output/impurity
    CalculateNodeOutput(nodes[0]);

    // stack of (node_index, depth)
    std::vector<std::pair<std::size_t,int>> idx_stack;
    idx_stack.emplace_back(0, 0);

    while (!idx_stack.empty()) {
        auto [idx, depth] = idx_stack.back(); idx_stack.pop_back();
        Node &node = nodes[idx];

        std::size_t n_samples = node.sampleIndices.size();
        if (node.isLeaf || n_samples < static_cast<std::size_t>(min_samples_split_) || node.impurity < min_impurity_split_ || depth >= max_depth_) {
            node.isLeaf = true;
            CalculateNodeOutput(node);
            continue;
        }

        Split bestSplit;
        bestSplit.featureIndex = -1;
        bestSplit.impurity = std::numeric_limits<double>::max();
        bestSplit.postSplit = -1;
        bestSplit.data_left.clear();
        bestSplit.data_right.clear();

        splitter_.samples_split_ = node.sampleIndices;
        splitter_.SplitNode(bestSplit);
        if (bestSplit.featureIndex == -1) {
            node.isLeaf = true;
            CalculateNodeOutput(node);
            continue;
        }
        node.featureIndex = bestSplit.featureIndex;
        node.threshold = bestSplit.threshold;

        Node leftChild;
        leftChild.isLeaf = false;
        leftChild.depth = depth + 1;
        leftChild.sampleIndices = bestSplit.data_left;

        Node rightChild;
        rightChild.isLeaf = false;
        rightChild.depth = depth + 1;
        rightChild.sampleIndices = bestSplit.data_right;

        std::size_t leftIdx = nodes.size();
        nodes.push_back(std::move(leftChild));
        std::size_t rightIdx = nodes.size();
        nodes.push_back(std::move(rightChild));

        // link parent to children
        node.left = leftIdx;
        node.right = rightIdx;

        // push children indices for further splitting
        idx_stack.emplace_back(rightIdx, depth + 1);
        idx_stack.emplace_back(leftIdx, depth + 1);
    }

    return tree;
}

void TreeBuilder::CalculateNodeOutput(Node &node)
{
    Criteria criteria(impurity_measure_);
    if (treeType_ == TreeType::Classification) {
        node.valuesClassification.clear();
        std::vector<double> labels;
        labels.reserve(node.sampleIndices.size());
        for (std::size_t si : node.sampleIndices) {
            double v = 0.0;
            if (si < subtrainData_->Y.size() && !subtrainData_->Y[si].empty()) v = subtrainData_->Y[si][0];
            node.valuesClassification.push_back(static_cast<int>(v));
            labels.push_back(v);
        }
        if (labels.empty()) {
            node.impurity = 0.0;
        } else {
            criteria.NodeImpurity(node.impurity, labels);
        }
    } else {
        node.valuesRegression.clear();
        std::vector<double> outputs;
        outputs.reserve(node.sampleIndices.size());
        for (std::size_t si : node.sampleIndices) {
            double v = 0.0;
            if (si < subtrainData_->Y.size() && !subtrainData_->Y[si].empty()) v = subtrainData_->Y[si][0];
            node.valuesRegression.push_back(v);
            outputs.push_back(v);
        }
        if (outputs.empty()) {
            node.impurity = 0.0;
        } else {
            criteria.NodeImpurity(node.impurity, outputs);
        }
    }
}
