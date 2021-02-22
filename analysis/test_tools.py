'''Shared functions between the regional variation and provider ranking tests'''
import pickle
from datetime import datetime as dt
import pandas as pd
from tqdm import tqdm

class TestTools:
    '''Shared functions between the regional variation and provider ranking tests'''
    def __init__(self, logger, graphs, models, code_converter):
        self.logger = logger
        self.log = logger.log
        self.graphs = graphs
        self.models = models
        self.code_converter = code_converter

    def check_claim_validity(self, indexed_data):
        '''confirm claims have not been reversed'''
        self.log("Checking patient claim validity")
        patients_to_check = indexed_data.loc[indexed_data["item_count"] != 1, "patient_id"].unique().tolist()
        patient_groups = indexed_data.groupby("patient_id")
        items_to_remove = []
        for patient_id in tqdm(patients_to_check):
            patient = patient_groups.get_group(patient_id)
            items_to_check = patient.loc[indexed_data["item_count"] != 1, "item_count"].unique(
            ).tolist()
            item_groups = patient[patient["computed_service_code"].isin(
                items_to_check)].groupby("computed_service_code")
            for _, item_group in item_groups:
                dos_groups = item_group.groupby("service_date")
                zero_date_indices = item_group.loc[item_group["item_count"] == 0, "index"].unique(
                ).tolist()
                items_to_remove.extend(zero_date_indices)

                neg_date = item_group.loc[item_group["item_count"]
                                          == -1, "service_date"].unique().tolist()
                for date in neg_date:
                    date_claims = dos_groups.get_group(date)
                    date_total = date_claims["item_count"].sum()
                    indices = date_claims["index"].tolist()
                    if date_total == 0:
                        items_to_remove.extend(indices)
                    elif date_total < 0:
                        raise ValueError(
                            f"Patient {patient_id} has unusual claim reversals")
                    else:
                        # need to come back and fix this, but is ok because number of claims per day is unimportant
                        mdvs = date_claims.loc[date_claims["item_count"]
                                               == -1, "index"].tolist()
                        items_to_remove.extend(mdvs)
                        # mdvs = date_claims["item_count"].tolist()
                        # ex = []
                        # for i in range(len(mdvs) -1):
                        #     if mdvs[i + 1] == -1:
                        #         if mdvs[i] == -1:
                        #             raise ValueError("Unexpected item_count behaviour")

                        #         ex.append(indices[i])
                        #         ex.append(indices[i+1])

        return indexed_data[~indexed_data["index"].isin(items_to_remove)]

    def create_state_model(self, params, state, mba_funcs, all_unique_items, group=None):
        '''Commands related to creation, graphing and saving of the state models'''
        rp = params
        if group is None:
            group = mba_funcs.group_data

        documents = mba_funcs.create_documents(group)
        self.log(f"{len(documents)} transactions in {self.code_converter.convert_state_num(state)}")
        self.log("Creating model")
        d = mba_funcs.create_model(all_unique_items, documents, rp.min_support)
        model_dict_csv = self.logger.get_file_path(f"state_{state}_model.csv")
        self.write_model_to_file(d, model_dict_csv)
        # remove no other item:
        if "No other items" in d:
            for k in d["No other items"]:
                if k not in d:
                    d[k] = {}

            d.pop("No other items")

        for k in d.keys():
            d[k].pop("No other items", None)

        name = f"patient_id_compu_state_{state}_graph"
        state_name = self.code_converter.convert_state_num(state)
        title = f'Connections between compu when grouped by patient_id and in state {state_name}'

        try:
            colour = rp.colour_only
        except AttributeError:
            colour = True

        if colour:
            formatted_d, attrs, legend = self.models.mba.colour_mbs_codes(d)
        else:
            formatted_d, attrs, legend = mba_funcs.convert_graph_and_attrs(d)

        model_name = self.logger.get_file_path(f"model_state_{state}.pkl")
        with open(model_name, "wb") as f:
            pickle.dump(formatted_d, f)

        attrs_name = self.logger.get_file_path(f"attrs_state_{state}.pkl")
        with open(attrs_name, "wb") as f:
            pickle.dump(attrs, f)

        legend_file = self.logger.get_file_path(f"Legend_{state}.png")
        # self.graphs.graph_legend(legend, legend_file, "Legend")

        # mba_funcs.create_graph(formatted_d, name, title, attrs, graph_style=rp.graph_style)
        # source, target = self.graphs.convert_pgv_to_hv(formatted_d)
        # not_chord = self.graphs.create_hv_graph(source, target)
        # self.graphs.save_hv_fig(not_chord, "hv_test")
        # circo_filename = self.logger.output_path / f"{state}_circos"
        # self.graphs.plot_circos_graph(formatted_d, attrs, circo_filename)
        self.graphs.create_visnetwork(formatted_d, name, title, attrs)
        # self.graphs.create_rchord(formatted_d,name,title)

        return d

    def write_model_to_file(self, d, filename):
        '''Save a graph model'''
        header = "Item is commonly claimed during unliateral joint replacements in the state " \
                  + "on the surgery date of service\n"
        with open(filename, 'w+') as f:
            f.write(header)
            subheader = "Category,Group,Sub-group,Item,Description\n"
            f.write(f'{subheader}')
            for node in d:
                line = self.code_converter.get_mbs_code_as_line(node)
                line = ','.join(line)
                f.write(f"{line}\n")

    def process_dataframe(self, params, data, surgeons_only=False, include_referrals_as_surgeon=False):
        '''process each dataframe before use'''
        rp = params
        patients_of_interest = data.loc[data["computed_service_code"] ==
                                        rp.code_of_interest, "patient_id"].unique().tolist()
        assert patients_of_interest
        patient_data = data[data["patient_id"].isin(patients_of_interest)]
        patient_data.reset_index(inplace=True)
        # patient_data = self.check_claim_validity(patient_data)
        # assert all(x >= 1 for x in patient_data["item_count"].tolist())

        # patient_data["patient_id"] = patient_data["patient_id"].astype(str)
        groups = patient_data.groupby("patient_id")
        final_data = pd.DataFrame(columns=patient_data.columns)
        exclusions = 0
        state_exclusions = 0
        splits = 0
        excess_patients = 0
        multiple_surgeons = 0
        for patient, group in tqdm(groups):
            dos = group.loc[group["computed_service_code"] == rp.code_of_interest, "service_date"].unique().tolist()
            number_of_surgeries = len(dos)
            if number_of_surgeries == 1:
                if surgeons_only:
                    surgeon_id = group.loc[(group["computed_service_code"] == rp.code_of_interest) & \
                        (group["service_date"] == dos[0]), "provider_id"].unique().tolist()
                    if len(surgeon_id) > 1:
                        multiple_surgeons += 1
                        self.log(f"Patient {patient} had multiple surgeons on date {dos[0]}; excluded")
                        continue

                    assert len(surgeon_id) == 1
                    if include_referrals_as_surgeon:
                        indices = group.loc[(group["service_date"] == dos[0]) & \
                            ((group["provider_id"] == surgeon_id[0]) | (group["reqref_prov"] == surgeon_id[0])), "index"].tolist()
                    else:
                        indices = group.loc[(group["service_date"] == dos[0]) & \
                            (group["provider_id"] == surgeon_id[0]), "index"].tolist()
                    assert len(indices) >= 1
                else:
                    indices = group.loc[group["service_date"] == dos[0], "index"].tolist()
                    assert len(indices) >= 1

                data_to_append = patient_data[patient_data["index"].isin(indices)]
                try:
                    exclude = rp.exclude_multiple_states
                except AttributeError:
                    exclude = False

                if exclude:
                    states = data_to_append['patient_state'].unique().tolist()
                    if len(states) > 1:
                        self.log(f"Patient {patient} had multiple states on date {dos[0]} and was excluded")
                        state_exclusions += 1
                        continue

                final_data = final_data.append(
                    data_to_append, ignore_index=True)
            elif number_of_surgeries == 0:
                self.log(
                    f"Patient {patient} has {len(dos)} claims for {rp.code_of_interest} and was excluded")
                exclusions += 1
                continue
            else:
                if number_of_surgeries > 2:
                    self.log(
                        f"Patient {patient} had {number_of_surgeries} surgeries")
                    excess_patients += number_of_surgeries - 2

                splits += 1
                for i, check_date in enumerate(dos):
                    if surgeons_only:
                        surgeon_id = group.loc[(group["computed_service_code"] == rp.code_of_interest) & \
                            (group["service_date"] == check_date), "provider_id"].unique().tolist()
                        if len(surgeon_id) > 1:
                            multiple_surgeons += 1
                            self.log(f"Patient {patient} had multiple surgeons on date {dos[0]}; excluded")
                            continue

                        assert len(surgeon_id) == 1
                        if include_referrals_as_surgeon:
                            indices = group.loc[(group["service_date"] == check_date) & \
                                ((group["provider_id"] == surgeon_id[0]) | (group["reqref_prov"] == surgeon_id[0])), "index"].tolist()
                        else:
                            indices = group.loc[(group["service_date"] == check_date) & \
                                (group["provider_id"] == surgeon_id[0]), "index"].tolist()

                        assert len(indices) >= 1
                    else:
                        indices = group.loc[group["service_date"] == check_date, "index"].tolist()
                        assert len(indices) >= 1

                    temp_df = patient_data[patient_data["index"].isin(indices)]
                    if rp.exclude_multiple_states:
                        states = temp_df['patient_idSTATE'].unique().tolist()
                        if len(states) > 1:
                            self.log(f"Patient {patient}_{i} had multiple states on date {check_date} and was excluded")
                            state_exclusions += 1
                            continue

                    temp_df["patient_id"] = temp_df["patient_id"] + f"_{i}"
                    final_data = final_data.append(temp_df, ignore_index=True)

        self.log(f"{len(groups)} starting patients for code {rp.code_of_interest}")
        self.log(f"{exclusions} patients excluded")
        self.log(f"{splits} patients split")
        self.log(f"{state_exclusions} exclusions after split")
        self.log(f"{multiple_surgeons} extra episodes excluded for multiple surgeons")
        assert len(final_data["patient_id"].unique()) == len(patients_of_interest) \
                                                  - exclusions \
                                                  + splits \
                                                  + excess_patients \
                                                  - state_exclusions \
                                                  - multiple_surgeons

        return final_data.drop(["index", "item_count"], axis=1)

    def get_test_data(self, data, code):
        '''process complete dataframe for the test'''
        # check if patients have multiple surgeries over multiple years
        patients = data["patient_id"].unique().tolist()
        original_no = len(patients)
        splits = 0
        additional_patients = 0
        for patient in tqdm(patients):
            dos = data.loc[data["patient_id"] == patient, "service_date"].unique().tolist()
            if len(dos) > 1:
                splits += 1
                additional_patients += len(dos) - 1
                for i, day in enumerate(dos):
                    data.loc[(data["patient_id"] == patient) & (data["service_date"] == day), "patient_id"] = f"{patient}__{i}"

        total_patients = len(data["patient_id"].unique())
        assert total_patients == original_no + additional_patients
        self.log(f"{splits} patients split")
        data_file = self.logger.get_file_path(f"test_data_{code}_{dt.now()}.pkl")
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)

        return data
