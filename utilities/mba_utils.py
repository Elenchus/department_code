'''Association analysis functions'''
import operator
import pandas as pd
from numpy import nan
from scipy.stats import fisher_exact

class MbaUtils:
    '''Class for association analysis'''
    filters: dict = {
        'confidence': {
            'operator': operator.ge,
            'value': 0
        },
        'conviction': {
            'operator': operator.ge,
            'value': 0
        },
        'lift': {
            'operator': operator.ge,
            'value': 0
        },
        'odds_ratio': {
            'operator': operator.ge,
            'value': 0
        },
        'certainty_factor': {
            'operator': operator.ge,
            'value': -1
        }
    }

    def __init__(self, code_converter, graphs):
        self.code_converter = code_converter
        self.graphs = graphs

    def update_filters(self, filters):
        '''Updates thresholds for interest measures'''
        for k, v in filters.items():
            if k not in self.filters:
                raise KeyError(f"Invalid filter {k}")
            for key, val in v.items():
                if key not in self.filters[k]:
                    raise KeyError(f"Invalid {k} filter option {key}")
                self.filters[k][key] = val

    def compare_transaction_to_model(self, items, model):
        '''compare the items in a single transaction to a graph dictionary model'''
        _items = {}
        for i in items:
            if i not in _items:
                _items[i] = {}

        diamonds = []
        for k in model.keys():
            if k in _items:
                for key in model[k].keys():
                    if key not in _items:
                        diamonds.append(key)
                        _items[k][key] = {'color': 'red'}
                    else:
                        _items[k][key] = None

        return _items, diamonds

    def find_repeated_abnormal_nodes(self, non_unique_basket, model, threshold=10):
        '''find nodes that occur repeatedly in the basket, but not in the model'''
        improper, _ = self.check_basket_for_presences(non_unique_basket, model)
        basket = list(set(non_unique_basket))
        diamonds = []
        for k in basket:
            if improper.get(k, -1) > threshold:
                diamonds.append(k)

        return diamonds

    def get_nodes_from_digraph(self, d):
        '''return set of nodes in a graph dictionary''' # this might be a repeat of GraphUtils.flatten_graph_dict
        a = set()
        for x in d.keys():
            a.add(x)
            for y in d[x].keys():
                a.add(y)

        return a

    def check_basket_for_absences(self, basket, model):
        '''find nodes in the model missing in the basket'''
        tally = 0
        for item in model.keys():
            if item in basket:
                for expected_item in model[item].keys():
                    if expected_item not in basket:
                        tally += 1
        return tally

    def check_basket_for_presences(self, basket, model, threshold=0):
        '''find nodes in the basket missing in the model'''
        # two problems - unique item differences, and repeated item differences
        tally = {i: 0 for i in set(basket)}
        nodes = self.get_nodes_from_digraph(model)
        for item in basket:
            if item in nodes:
                tally[item] -= 1
            else:
                tally[item] += 1

        proper = {}
        improper = {}
        for k, v in tally.items():
            x = proper if v < threshold + 1 else improper
            x[k] = abs(v)

        return improper, proper

    def color_providers(self, d, data, colour_keys=True, colour_vals=True):
        '''colour nodes based on MBS SPR_RSP'''
        def get_provider_val(spr):
            spr = int(spr)
            rows = data[data['SPR'] == spr]
            rsps = rows['SPR_RSP'].mode().tolist()
            if len(rsps) == 1:
                rsp = rsps[0]
            else:
                rsp = 'Multiple'

            return rsp

        lookup = {}
        for k in d.keys():
            lookup[k] = get_provider_val(k)

        used_colors = set()
        for k, v in d.items():
            if colour_keys:
                if lookup[k] not in lookup:
                    color = get_provider_val(k)
                    lookup[k] = color
                    used_colors.add(color)
            if colour_vals:
                for key in v.keys():
                    if key not in lookup:
                        color = get_provider_val(key)
                        lookup[key] = color
                        used_colors.add(color)

        colour_table = {}
        for i, col in enumerate(used_colors):
            color = int(i * 255 / len(used_colors))
            anti_col = 255 - color
            # g = int(min(color, anti_col)/2)
            c = '{:02x}'.format(color)
            a = '{:02x}'.format(anti_col)

            colour_table[str(col)] = {'color': f"#{a}{c}{0}"}

        colors = {}
        for k, v in lookup.items():
            colors[str(k)] = colour_table[str(v)]

        return colors, colour_table

    def colour_mbs_codes(self, d):
        '''colour nodes based on MBS item groups'''
        get_color = {
            'I': 'red', # item not in dictionary. Colours: https://serialmentor.com/dataviz/color-pitfalls.html
            '1': '#E69F00',
            '2': '#56B4E9',
            '3': '#009E73',
            '4': '#F0E442',
            '5': '#0072B2',
            '6': '#D55E00',
            '7': '#CC79A7',
            '8': '#ffd912'
        }

        all_items = self.graphs.flatten_graph_dict(d)
        attrs = {}
        color_map = set()
        for item in all_items:
            if item == "No other items":
                group_no = 'I'
            else:
                group_no = self.code_converter.convert_mbs_code_to_group_numbers(item)[0]
                if len(group_no) == 1:
                    group_no = 'I'

            color = get_color[group_no]
            attrs[item] = {'color': color}
            color_map.add(group_no)

        legend = {}
        for color in color_map:
            color_name = get_color[color]
            color_label = self.code_converter.convert_mbs_category_number_to_label(color)
            legend[color_name] = {'color': color_name,
                                  'label': color_label.replace(' ', '\n'),
                                  'labeljust': ';', 'rank': 'max'}

        return (d, attrs, legend)

    def convert_mbs_codes(self, d):
        '''change node text from MBS item codes to human readable descriptions'''
        get_color = {
            'I': 'red', # for item not in dictionary
            '1': '#E69F00',
            '2': '#56B4E9',
            '3': '#009E73',
            '4': '#F0E442',
            '5': '#0072B2',
            '6': '#D55E00',
            '7': '#CC79A7',
            '8': '#ffd912'
        }

        lookup = {}
        for k in d.keys():
            if k == "No other items":
                lookup["No other items"] = "No other items"
            else:
                labels = self.code_converter.convert_mbs_code_to_group_labels(k)
                lookup[k] = '\n'.join(labels)

        new_data = {}
        colors = {}
        color_map = set()
        for k, v in d.items():
            new_k = f'{lookup[k]}\n{k}'
            if new_k not in new_data:
                if new_k == "No other items" or new_k == "No other items\nNo other items":
                    group_no = 'I'
                else:
                    group_no = self.code_converter.convert_mbs_code_to_group_numbers(k)[0]

                color = get_color[group_no]
                colors[new_k] = {'color': color}
                color_map.add(group_no)
                new_data[new_k] = {}
            for key, val in v.items():
                if key not in lookup:
                    if k == "No other items":
                        lookup["No other items"] = "No other items"
                    else:
                        labels = self.code_converter.convert_mbs_code_to_group_labels(key)
                        lookup[key] = '\n'.join(labels)

                new_key = f'{lookup[key]}\n{key}'
                new_data[new_k][new_key] = val

                if key not in d:
                    if new_key == "No other items" or new_key == "No other items\nNo other items":
                        group_no = 'I'
                    else:
                        group_no = self.code_converter.convert_mbs_code_to_group_numbers(key)[0]

                    color = get_color[group_no]
                    colors[new_key] = {'color': color}
                    color_map.add(group_no)

        legend = {}
        for color in color_map:
            color_name = get_color[color]
            color_label = self.code_converter.convert_mbs_category_number_to_label(color)
            legend[color_name] = {'color': color_name,
                                  'label': color_label.replace(' ', '\n'),
                                  'labeljust': ';',
                                  'rank': 'max'}

        return (new_data, colors, legend)

    def convert_rsp_keys(self, d):
        '''convert node text from SPR_RSP numbers to human readable text'''
        lookup = {}
        for k in d.keys():
            if k == "No other items":
                lookup["No other items"] = "No other items"
            else:
                lookup[k] = self.code_converter.convert_rsp_num(k)

        new_data = {}
        for k, v in d.items():
            if lookup[k] not in new_data:
                new_data[lookup[k]] = {}
            for key, val in v.items():
                if key not in lookup:
                    if key == "No other items":
                        lookup["No other items"] = "No other items"
                    else:
                        lookup[key] = self.code_converter.convert_rsp_num(key)

                new_data[lookup[k]][lookup[key]] = val

        return new_data

    def exception_rules(self, antecedent, consequent, threshold, documents):
        '''Find exception rules for an item pair'''
        X_subset = []
        item_subset = {}
        exclusions = []
        for doc in documents:
            if antecedent in doc:
                X_subset.append(doc)
                for item in doc:
                    item_subset[item] = item_subset.get(item, 0) + 1

        support_Y = item_subset[consequent] / len(X_subset)
        for item in list(item_subset.keys()):
            if item == consequent:
                continue

            support_X = item_subset[item] / len(X_subset)
            support_XY = 0
            for doc in X_subset:
                if consequent in doc and item in doc:
                    support_XY += 1

            support_XY = support_XY / len(X_subset)
            confidence = support_XY / support_X
            num = (1 - support_Y)
            den = (1 - confidence)
            conviction = num / den if den != 0 else 2 * threshold
            if conviction < threshold:
                exclusions.append(item)

        return exclusions

    def pairwise_market_basket(self,
                               items,
                               documents,
                               min_support=0.1,
                               max_p_value=1,
                               absolute_min_support_count=0):
        '''find association rules between item pairs'''
        group_len = len(documents)
        if min_support < 1:
            min_occurrence = min_support * group_len
        else:
            min_occurrence = min_support

        if min_occurrence < absolute_min_support_count:
            min_occurrence = absolute_min_support_count

        reduced_items = {"No other items": 0}
        for item in items:
            reduced_items[item] = 0
        for doc in documents:
            for item in doc:
                reduced_items[item] += 1

        keys = list(reduced_items.keys())
        for item in keys:
            if reduced_items[item] < min_occurrence:
                reduced_items.pop(item)

        reduced_item_list = reduced_items.keys()
        counts = pd.DataFrame(0, index=reduced_item_list, columns=reduced_item_list)
        for doc in documents:
            for item in doc:
                if item not in reduced_item_list:
                    continue

                for item_2 in doc:
                    if item_2 not in reduced_item_list:
                        continue

                    counts.at[item, item_2] += 1

        # row_list = []
        d = {}
        for a in reduced_item_list:
            for b in reduced_item_list:
                if a == b:
                    continue

                count = counts.at[a, b]
                if  count >= min_occurrence:
                    f11 = count
                    f10 = reduced_items[a] - f11
                    f01 = reduced_items[b] - f11
                    f00 = group_len - (f10 + f01 + count)
                    odds_ratio, p_value = fisher_exact([[f11, f10], [f01, f00]], alternative='greater')
                    if odds_ratio is nan:
                        odds_ratio = 9999

                    if p_value > max_p_value:
                        continue

                    support = count / group_len
                    support_a = reduced_items[a] / group_len
                    support_b = reduced_items[b] / group_len

                    lift = support / (support_a * support_b)
                    confidence = support / support_a

                    conviction = (1 - support_b) / (1 - confidence) if confidence != 1 else 9999
                    if confidence > support_b:
                        certainty_factor = (confidence - support_b) / (1 - support_b) if support_b != 1 else 9999
                    elif confidence < support_b:
                        certainty_factor = (confidence - support_b) / support_b if support_b != 0 else 9999
                    else:
                        certainty_factor = 0

                    for k, v in self.filters.items():
                        comp = v['operator']
                        val = v['value']
                        if k == 'lift':
                            fil = lift
                        elif k == 'confidence':
                            fil = confidence
                        elif k == 'conviction':
                            fil = conviction
                        elif k == 'odds_ratio':
                            fil = odds_ratio
                        elif k == 'certainty_factor':
                            fil = certainty_factor
                        else:
                            raise KeyError(f"No matching association rule {k}")

                        if not comp(fil, val):
                            break
                    else:
                        if a not in d:
                            d[a] = {}

                        d[a][b] = {"weight": confidence}

        return d
