'''functions for quick graphing'''
import pickle
from copy import deepcopy
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()

class GraphUtils():
    '''Functions for visualising, saving, and manipulating plots and graphs'''
    def __init__(self, logger):
        self.logger = logger

    def basic_histogram(self, data, filename):
        '''creates and saves a histogram'''
        self.logger.log("Plotting histogram")
        n_bins = len(set(data))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(data, n_bins)
        self.save_plt_fig(fig, filename)

    def categorical_plot_group(self, x, y, legend_labels, title, filename, axis_labels=None):
        '''creates and saves a categorical plot'''
        self.logger.log(f"Plotting bar chart: {title}")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        assert len(x) == len(y)
        for i, _ in enumerate(y):
            ax.scatter([str(q) for q in x[i]], y[i], label=legend_labels[i])

        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ttl = fig.suptitle(title)
        if axis_labels is not None:
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])

        self.save_plt_fig(fig, filename, (lgd, ttl, ))

    @staticmethod
    def convert_adjacency_matrix_to_graph(a_m):
        '''convert a pandas format adjacency matrix to a graph dictionary'''
        items = a_m.columns
        graph = {}
        for ante in items:
            for con in items:
                val = a_m.at[ante, con]
                if val != 0:
                    if ante in graph:
                        graph[ante][con] = None
                    else:
                        graph[ante] = {con: None}

        return graph

    def convert_graph_to_adjacency_matrix(self, graph):
        '''convert a graph dictionary to an adjacency matrix in pandas format'''
        items = [x for x in self.flatten_graph_dict(graph)]
        a_m = pd.DataFrame(0, index=items, columns=items)
        for ante in graph.keys():
            for con in graph[ante].keys():
                a_m.at[ante, con] += 1

        return a_m

    @staticmethod
    def convert_pgv_to_simple(graph):
        '''convert a pygraphviz format dictionary to a simpler format'''
        ng = {}
        for k, v in graph.items():
            ng[k] = set(v.keys())

        return ng

    @staticmethod
    def convert_simple_to_pgv(graph):
        '''convert a simple graph dictionary to pygraphviz format'''
        ng = {}
        for k, v in graph.items():
            ng[k] = {x: {} for x in v}

        return ng

    def create_boxplot(self, data, title, filename):
        '''creates and saves a single boxplot'''
        self.logger.log(f"Plotting boxplot: {title}")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(data)
        fig.suptitle(title)
        self.save_plt_fig(fig, filename)

    def create_boxplot_group(self, data, labels, title, filename, axis_labels=None):
        '''creates and saves a group of boxplots'''
        self.logger.log(f"Plotting boxplot group: {title}")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(data)
        fig.suptitle(title)
        ax.set_xticklabels(labels)
        if axis_labels is not None:
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])

        self.save_plt_fig(fig, filename)

    def create_feature_matrix_from_graph(self, graph):
        '''create a graph feature matrix in pandas format from a graph dictionary'''
        idx = self.flatten_graph_dict(graph)
        mat = pd.DataFrame(0, index=idx, columns=idx)
        for item in idx:
            mat.at[item, item] = 1

        return mat

    def create_grouped_barchart(self, data, bar_labels, group_labels, title, filename, axis_labels):
        '''create several bar charts in one graph from a list of lists'''
        bar_width = 0.25
        r = []
        r.append(np.arange(len(data[0])))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, _ in enumerate(data):
            ax.bar(r[-1], data[i], width=bar_width, label=bar_labels[i])
            r.append([x + bar_width for x in r[-1]])

        ax.set_xticks([r + bar_width for r in range(len(group_labels))])
        ax.set_xticklabels(group_labels, rotation=45, ha="right")
        if axis_labels is not None:
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])

        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ttl = fig.suptitle(title)
        self.save_plt_fig(fig, filename, (lgd, ttl, ))


    def create_rchord(self, graph, name, title):
        '''create a chord diagram from a graph dictionary'''
        from_nodes = []
        to_nodes = []
        values = []
        for key, val in graph.items():
            for k in val.keys():
                from_nodes.append(key)
                to_nodes.append(k)
                values.append(10)

        edges = pd.DataFrame([from_nodes, to_nodes, values])
        edges = edges.transpose()
        edges.columns = ['from', 'to', 'value']

        aplot = importr('graphics')
        circlize = importr('circlize')
        rDevices = importr('grDevices')
        r_am = pandas2ri.py2rpy(edges)
        filename = self.logger.output_path / f'{name}.png'
        rDevices.png(str(filename), width=800, height=800)
        circlize.chordDiagram(r_am,
                              directional=1,
                              direction_type="arrows",
                              link_arr_type="big.arrow")
        aplot.title(title, cex=0.8)
        rDevices.dev_off()

    def create_scatter_plot(self, data, labels, title, filename, legend_names=None, axis_labels=None):
        '''creates and saves a scatter plot'''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        scatter = ax.scatter(data[:, 0], data[:, 1], c=labels)
        if legend_names is not None:
            handles, _ = scatter.legend_elements(num=None)
            legend = ax.legend(handles,
                               legend_names,
                               loc="upper left",
                               title="Legend",
                               bbox_to_anchor=(1, 0.5))

        if axis_labels is not None:
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])

        ttl = fig.suptitle(title)

        filename = self.logger.output_path / filename

        if legend_names is None:
            self.save_plt_fig(fig, filename, [ttl])
        else:
            self.save_plt_fig(fig, filename, [ttl, legend])

    def create_visnetwork(self, graph, name, title, attrs=None):
        '''Create and save a visnetwork graph from a graph dictionary'''
        if attrs:
            nodes = pd.DataFrame(attrs)
            nodes = nodes.transpose()
            nodes['id'] = nodes.index
            nodes['label'] = nodes.index
            nodes['groupname'] = nodes['color']
        else:
            nodes = pd.DataFrame(self.flatten_graph_dict(graph), columns=['id'])
            nodes['label'] = nodes['id']

        from_nodes = []
        to_nodes = []
        colors = []
        for key, val in graph.items():
            for k in val.keys():
                from_nodes.append(key)
                to_nodes.append(k)
                color = graph[key][k].get('color', 'black')
                colors.append(color)

        edges = pd.DataFrame([from_nodes, to_nodes, colors])
        edges = edges.transpose()
        edges.columns = ['from', 'to', 'color']

        vn = importr('visNetwork')
        r_nodes = pandas2ri.py2rpy(nodes)
        r_edges = pandas2ri.py2rpy(edges)

        net = vn.visNetwork(r_nodes, r_edges, main=title, width="100%", improvedLayout=False)
        net = vn.visEdges(net, arrows='to')
        net = vn.visNodes(net, shape='circle', widthConstraint=50)
        # net = vn.visLegend(net)

        vispath = self.logger.output_path / f"{name}"
        vishtml = f"{vispath}.html"
        vn.visSave(net, vishtml)
        # vispng = f"{vispath}.png"
        # imgkit.from_file(vishtml, vispng)

    def find_graph_components(self, graph):
        '''find separate graph components'''
        am = self.convert_graph_to_adjacency_matrix(graph)
        keys = am.index.tolist()
        ITEMS = am.index.tolist()
        def search_matrix(node, current_component, connected_nodes):
            for item in ITEMS:
                row = am.at[node, item]
                col = am.at[item, node]
                if (row > 0 or col > 0) and item not in current_component:
                    current_component.add(item)
                    connected_nodes.add(item)

        identified = set()
        components = []
        while keys:
            component = set()
            connected_nodes = set()
            node = keys.pop()
            if node in identified:
                continue

            component.add(node)
            while True:
                search_matrix(node, component, connected_nodes)
                if not connected_nodes:
                    break

                node = connected_nodes.pop()

            identified.update(component)
            components.append(component)

        return components

    @staticmethod
    def flatten_graph_dict(dictionary):
        ''' Returns a set of all keys and values in a graph dictionary'''
        temp = set()
        for k, v in dictionary.items():
            temp.add(k)
            for key in v.keys():
                temp.add(key)

        return temp

    def graph_component_finder(self, graph):
        '''find separate graph components'''
        am = self.convert_graph_to_adjacency_matrix(graph)
        keys = am.index.tolist()
        ITEMS = am.index.tolist()
        def search_matrix(node, current_component, connected_nodes):
            for item in ITEMS:
                row = am.at[node, item]
                col = am.at[item, node]
                if (row > 0 or col > 0) and item not in current_component:
                    current_component.add(item)
                    connected_nodes.add(item)

        identified = set()
        components = []
        while keys:
            component = set()
            connected_nodes = set()
            node = keys.pop()
            if node in identified:
                continue

            component.add(node)
            while True:
                search_matrix(node, component, connected_nodes)
                if not connected_nodes:
                    break

                node = connected_nodes.pop()

            identified.update(component)
            components.append(component)

        return components

    def graph_edit_distance(self, expected, test, attrs=None, edge_distance_costs=False, split_missing_and_unexpected=True):
        '''get the graph edit distance between two graphs using MBS item fees if available'''
        if not test: # this is used to ignore providers with no associated claims
            if split_missing_and_unexpected:
                return (0, 0), {}, {}
            else:
                return 0, {}, {}

        expected = self.stringify_graph(expected)
        test = self.stringify_graph(test)

        unexpected_score = 0
        missing_score = 0
        d = {x: {} for x in self.flatten_graph_dict(test)}
        if attrs is None:
            attrs = {x: {} for x in d}

        possible_nodes = list(x for x in self.flatten_graph_dict(expected))
        keys = list(d.keys())
        edit_attrs = {}
        edit_history = deepcopy(test)
        for key in keys:
            if key not in possible_nodes:
                if key in attrs:
                    unexpected_score += attrs[key].get('weight', 1) # fee
                else:
                    unexpected_score += 1

                if key in test:
                    edges = test[key]
                    if edge_distance_costs:
                        unexpected_score += sum([test[key][x].get('weight', 1) for x in edges])# confidence

                    for k in edges:
                        edit_history[key][k]['color'] = '#D55E00'

                edit_attrs[key] = {'shape': 'database'}
                # edit_attrs[key] = {'shape': 'house'}
                d.pop(key)
            else:
                edit_attrs[key] = {'shape': 'circle'}

        nodes_to_add = set()
        edges_to_add = {}
        for key in d.keys():
            if key not in expected:
                continue

            possible_edges = set(x for x in expected[key].keys())
            if key in test:
                actual_edges = set(test[key].keys())
            else:
                actual_edges = set()

            missing_edges = possible_edges - actual_edges
            should_have = possible_edges.intersection(actual_edges)
            should_have.update(missing_edges)
            should_not_have = actual_edges - possible_edges
            d[key] = {x: {} for x in should_have}
            missing_nodes = set()
            for node in missing_edges:
                if node not in d:
                    missing_nodes.add(node)

            edges_to_add[key] = missing_edges

            nodes_to_add.update(missing_nodes)
            if edge_distance_costs:
                missing_score += sum([expected[key][x].get('weight', 1) for x in missing_edges])
                unexpected_score += sum([test[key][x].get('weight', 1) for x in should_not_have])# confidence

            for k in should_not_have:
                edit_history[key][k]['color'] = '#D55E00'

        for key in edges_to_add:
            for k in edges_to_add[key]:
                if key not in edit_history:
                    edit_history[key] = {}

                edit_history[key][k] = {'color': '#F0E442'}

        while nodes_to_add:
            ignore_list = []
            node = nodes_to_add.pop()
            if node in attrs:
                missing_score += attrs[node].get('weight', 1) # fee
            else:
                missing_score += 1

            edit_attrs[node] = {'shape': 'box'}
            # edit_attrs[node] = {'shape': 'invhouse'}

            if node not in expected:
                ignore_list.append(node)
                continue


            edges = expected[node]
            if edge_distance_costs:
                missing_score += sum([expected[node][x].get('weight', 1) for x in edges]) # confidence

            d[node] = edges
            edit_history[node] = edges
            for k in edit_history[node]:
                edit_history[node][k]['color'] = '#F0E442'

            for new_node in edges:
                if new_node not in d and new_node not in ignore_list:
                    nodes_to_add.add(new_node)

        if split_missing_and_unexpected:
            score = (unexpected_score, missing_score)
        else:
            score = unexpected_score + missing_score

        return score, edit_history, edit_attrs

    def save_plt_fig(self, fig, filename, bbox_extra_artists=None):
        '''Save a plot figure to file with timestamp'''
        current = datetime.now().strftime("%Y%m%dT%H%M%S")
        output_path = self.logger.get_file_path(f"{filename}_{current}")
        pickle_path = self.logger.get_file_path(f"{output_path}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(fig, f)

        if bbox_extra_artists is None:
            fig.savefig(output_path)
        else:
            fig.savefig(output_path, bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')

        plt.close(fig)

    @staticmethod
    def stringify_graph(graph):
        '''Ensures all graph dictionary keys and values are in string format'''
        str_graph = {}
        for key in graph:
            str_graph[str(key)] = {}
            for k in graph[key]:
                str_graph[str(key)][str(k)] = graph[key][k]

        return str_graph
