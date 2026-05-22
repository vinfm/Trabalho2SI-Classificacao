#include "Splitter.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

Splitter::Splitter(CriterionType criterionType)
	: min_samples_leaf_(1),
	  max_features_(0),
	  gen_(nullptr),
	  shuffle_features_(false),
	  n_samples_total_(0),
	  n_features_(0),
	  criteria_(criterionType),
	  criterionType_(criterionType)
{
}

Splitter::Splitter(CriterionType criterionType,
				   const std::vector<std::vector<double>> &data,
				   const std::vector<std::vector<double>> &output_data,
				   std::size_t n_features,
				   std::size_t max_features,
				   int min_samples_leaf)
	: data_(data),
	  output_data_(output_data),
	  min_samples_leaf_(min_samples_leaf),
	  max_features_(max_features),
	  gen_(nullptr),
	  shuffle_features_(max_features < n_features),
	  n_samples_total_(data.size()),
	  n_features_(n_features),
	  criteria_(criterionType),
	  criterionType_(criterionType)
{
	feature_order_.resize(n_features_);
	std::iota(feature_order_.begin(), feature_order_.end(), 0);
}

Split::Split()
	: featureIndex(-1), threshold(0.0), impurity(std::numeric_limits<double>::max()), gain(-std::numeric_limits<double>::infinity()), postSplit(-1)
{
}

void Splitter::SplitNode(Split &split)
{
	split.featureIndex = -1;
	split.threshold = 0.0;
	split.impurity = std::numeric_limits<double>::max();
	split.gain = -std::numeric_limits<double>::infinity();
	split.postSplit = -1;
	split.data_left.clear();
	split.data_right.clear();

	if (samples_split_.empty() || data_.empty() || n_features_ == 0) {
		return;
	}

	if (shuffle_features_ && max_features_ < n_features_) {
		if (gen_) {
			std::shuffle(feature_order_.begin(), feature_order_.end(), *gen_);
		}
	}

	std::vector<double> parent_targets;
	parent_targets.reserve(samples_split_.size());
	for (std::size_t sampleIndex : samples_split_) {
		if (sampleIndex < output_data_.size() && !output_data_[sampleIndex].empty()) {
			parent_targets.push_back(output_data_[sampleIndex][0]);
		}
	}

	if (parent_targets.size() < 2) {
		return;
	}

	const double parent_impurity = criteria_.calculateImpurity(parent_targets);
	const std::size_t total_samples = parent_targets.size();

	double best_gain = -std::numeric_limits<double>::infinity();

	std::size_t features_to_check = shuffle_features_ ? max_features_ : n_features_;
	for (std::size_t idx = 0; idx < features_to_check && idx < feature_order_.size(); ++idx) {
		int featureIndex = feature_order_[idx];

		std::vector<std::pair<double, std::size_t>> feature_values;
		feature_values.reserve(samples_split_.size());
		for (std::size_t sampleIndex : samples_split_) {
			if (sampleIndex < data_.size() && static_cast<std::size_t>(featureIndex) < data_[sampleIndex].size()) {
				feature_values.emplace_back(data_[sampleIndex][featureIndex], sampleIndex);
			}
		}

		if (feature_values.size() < 2) {
			continue;
		}

		std::sort(feature_values.begin(), feature_values.end(),
				  [](const auto &a, const auto &b) { return a.first < b.first; });

		for (std::size_t i = 1; i < feature_values.size(); ++i) {
			if (feature_values[i - 1].first == feature_values[i].first) {
				continue;
			}

			const std::size_t left_count = i;
			const std::size_t right_count = feature_values.size() - i;
			if (left_count < static_cast<std::size_t>(min_samples_leaf_) || right_count < static_cast<std::size_t>(min_samples_leaf_)) {
				continue;
			}

			std::vector<double> left_targets;
			std::vector<double> right_targets;
			left_targets.reserve(left_count);
			right_targets.reserve(right_count);

			for (std::size_t j = 0; j < i; ++j) {
				const std::size_t sampleIndex = feature_values[j].second;
				if (sampleIndex < output_data_.size() && !output_data_[sampleIndex].empty()) {
					left_targets.push_back(output_data_[sampleIndex][0]);
				}
			}
			for (std::size_t j = i; j < feature_values.size(); ++j) {
				const std::size_t sampleIndex = feature_values[j].second;
				if (sampleIndex < output_data_.size() && !output_data_[sampleIndex].empty()) {
					right_targets.push_back(output_data_[sampleIndex][0]);
				}
			}

			if (left_targets.empty() || right_targets.empty()) {
				continue;
			}

			const double left_impurity = criteria_.calculateImpurity(left_targets);
			const double right_impurity = criteria_.calculateImpurity(right_targets);
			const double weighted_impurity = (static_cast<double>(left_targets.size()) / static_cast<double>(total_samples)) * left_impurity
										   + (static_cast<double>(right_targets.size()) / static_cast<double>(total_samples)) * right_impurity;
			const double gain = parent_impurity - weighted_impurity;

			if (gain > best_gain) {
				best_gain = gain;
				split.featureIndex = featureIndex;
				split.threshold = (feature_values[i - 1].first + feature_values[i].first) / 2.0;
				split.impurity = weighted_impurity;
				split.gain = gain;
				split.postSplit = static_cast<int>(i);
				split.data_left.clear();
				split.data_right.clear();
				for (std::size_t j = 0; j < i; ++j) {
					split.data_left.push_back(feature_values[j].second);
				}
				for (std::size_t j = i; j < feature_values.size(); ++j) {
					split.data_right.push_back(feature_values[j].second);
				}
			}
		}
	}
}
