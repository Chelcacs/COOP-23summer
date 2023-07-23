from brownie import FELToken, ProjectContract, accounts, config
from sklearn.linear_model import LinearRegression
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from felt.builder import upload_model
import sys
sys.path.append(r'/Users/a123/Desktop/coop/COOP-23summer/federated-learning-token/tabnet_pre')
# import data_and_model as dm
import CusModel as dm


def create_plan(project, builder):
    ## DEFINE MODEL ###
    # model = LinearRegression() chel
    # Network parameters
    model = dm.get_model()
    cid = upload_model(model)

    ### PROVIDE REWARDS AND UPLOAD PLAN ###
    FELToken[-1].increaseAllowance(project.address, 1000, {"from": builder})
    project.createPlan(cid, 3, 10, {"from": builder})


def main():
    project = ProjectContract[-1]
    builder = accounts.add(config["wallets"]["owner_key"])
    create_plan(project, builder)
