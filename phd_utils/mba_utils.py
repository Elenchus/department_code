import operator
import pandas as pd
from enum import auto, Enum
from scipy.stats import fisher_exact

class MbaUtils:
    filters:dict = {
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
        }
    }

    def __init__(self, code_converter):
        self.code_converter=code_converter

    def update_filters(self, filters):
        for k, v in filters.items():
            if k not in self.filters:
                raise KeyError(f"Invalid filter {k}")
            for key, val in v.items():
                if key not in self.filters[k]:
                    raise KeyError(f"Invalid {k} filter option {key}")
                self.filters[k][key] = val

    def compare_transaction_to_model(self, items, model):
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
        improper, _ = self.check_basket_for_presences(non_unique_basket, model)
        basket = list(set(non_unique_basket))
        diamonds = []
        for k in basket:
            if improper.get(k, -1) > threshold:
                diamonds.append(k)

        return diamonds

    def find_underutilised_normal_nodes(self, items, interest, threshold=10):
        diamonds = []
        for k in items:
            count = interest.get(k, -1)  
            raise KeyError
            if count == 1:
                diamonds.append(k)

        return diamonds

    def get_nodes_from_digraph(self, d):
        a = set()
        for x in d.keys():
            a.add(x)
            for y in d[x].keys():
                a.add(y)

        return a

    def check_basket_for_absences(self, basket, model):
        tally = 0
        for item in model.keys():
            if item in basket:
                for expected_item in model[item].keys():
                    if expected_item not in basket:
                        tally += 1
        return tally

    def check_basket_for_presences(self, basket, model):
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
            x = proper if v < 0 else improper
            x[k] = abs(v)

        return improper, proper
            
    def color_providers(self, d, data):
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
            if lookup[k] not in lookup:
                color = get_provider_val(k)
                lookup[k] = color
                used_colors.add(color)
            for key in v.keys():
                if key not in lookup:
                    color = get_provider_val(key)
                    lookup[key] = color
                    used_colors.add(color)

        colour_table = {}
        for i, col in enumerate(used_colors):
            color = int(i * 255 / len(used_colors))
            anti_col = 255 - color
            g = int(min(color, anti_col)/2)
            c = '{:02x}'.format(color)
            a = '{:02x}'.format(anti_col)

            colour_table[col] = {'color': f"#{a}{c}{0}"}

        colors = {}
        for k, v in lookup.items():
            colors[k] = colour_table[v]

        return colors, colour_table

    def compare_items_to_model(self, items, model):
        pass

    def convert_mbs_codes(self, d):
        get_color = {
            'I': 'tomato', # for item not in dictionary
            '1': 'blue',
            '2': 'green',
            '3': 'red',
            '4': 'yellow',
            '5': 'cyan',
            '6': 'khaki',
            '7': 'orange',
            '8': 'darkorchid' 
        }

        lookup = {}
        for k in d.keys():
            labels = self.code_converter.convert_mbs_code_to_group_labels(k)
            lookup[k] = '\n'.join(labels)

        new_data = {}
        colors = {}
        color_map = set()
        for k, v in d.items():
            new_k = f'{lookup[k]}\n{k}'
            if new_k not in new_data:
                group_no = self.code_converter.convert_mbs_code_to_group_numbers(k)[0]
                color = get_color[group_no]
                colors[new_k] = {'color': color}
                color_map.add(group_no)
                new_data[new_k] = {}
            for key, val in v.items():
                if key not in lookup:
                    labels = self.code_converter.convert_mbs_code_to_group_labels(key)
                    lookup[key] = '\n'.join(labels)

                new_key = f'{lookup[key]}\n{key}'
                new_data[new_k][new_key] = val

        legend = {}
        for color in color_map:
            color_name = get_color[color]
            color_label = self.code_converter.convert_mbs_category_number_to_label(color)
            legend[color_name] = {'color': color_name, 'label': color_label, 'labeljust': ';', 'rank': 'max'}

        return (new_data, colors, legend)

    def convert_rsp_keys(self, d):
        lookup = {}
        for k in d.keys():
            lookup[k] = self.code_converter.convert_rsp_num(k)

        new_data = {}
        for k, v in d.items():
            if lookup[k] not in new_data:
                new_data[lookup[k]] = {}
            for key, val in v.items():
                if key not in lookup:
                    lookup[key] = self.code_converter.convert_rsp_num(key)
                new_data[lookup[k]][lookup[key]] = val

        return new_data

    def pairwise_market_basket(self,
                                items,
                                documents,
                                min_support = 0.1,
                                max_p_value=1):
        group_len = len(documents)
        if min_support < 1:
            min_occurrence = min_support * group_len
        else:
            min_occurrence = min_support

        reduced_items = {}
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

                    if p_value > max_p_value:
                        continue
                    
                    support = count / group_len
                    support_a = reduced_items[a] / group_len
                    support_b = reduced_items[b] / group_len

                    lift = support / (support_a * support_b)
                    confidence = support / support_a

                    conviction = (1 - support_b) / (1 - confidence) if confidence != 1 else 9999

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
                        else:
                            raise KeyError(f"No matching association rule {k}")

                        if not comp(fil, val):
                            break
                    else:
                        if a not in d:
                            d[a] = {}

                        d[a][b] = None

                    # new_row = {"Antecedent": a, "Consequent": b, "Count": count, "Support": support, "Confidence": confidence, "Conviction": conviction, "Lift": lift, "Odds ratio": odds_ratio}
                    # row_list.append(new_row)

        # output = pd.DataFrame(row_list)

        return d