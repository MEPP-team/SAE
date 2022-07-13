import os
import random
import polyscope.imgui as psim
import torch
import polyscope as ps

from train import load_trainer
from models.utils.error_meshes import error_meshes_mm
from models.utils.get_coeffs import get_coeffs
from models.utils.colors import colors


def register_surface(array, name, triv, index_color, transparency=1.0):
    mesh = ps.register_surface_mesh(name, array.cpu().numpy(), triv)
    mesh.set_color(tuple(int(colors[index_color][i:i + 2], 16) / 255.0 for i in (1, 3, 5)))
    mesh.set_smooth_shade(True)
    mesh.set_transparency(transparency)
    return mesh


def reconstruction(wanted_index, opt, dataset, saturation=0.05):
    input, vertices = get_coeffs(wanted_index, opt, dataset)

    output = opt['model'](input)

    output_spatial = torch.matmul(opt['evecs'], output)

    if vertices is None:
        vertices = torch.matmul(opt['evecs'], input)

    register_surface(vertices[0], "Input", opt['TRIV'], 0)
    output_mesh = register_surface(output_spatial[0], "Output", opt['TRIV'], 1)

    distances = error_meshes_mm(vertices, output_spatial)[0]

    output_mesh.add_scalar_quantity(
        "Errors",
        distances.cpu().numpy(),
        enabled=True,
        cmap='reds',
        vminmax=(0, saturation)
    )

    print('Mean distance: %.2fmm' % (distances.mean().item() * 1000), end="\n\n")

    return opt["model"].enc(input)


def interp(wanted_indices, opt, dataset):
    a = 0.5
    input_0, vertices_0 = get_coeffs(wanted_indices[0], opt, dataset)
    input_1, vertices_1 = get_coeffs(wanted_indices[1], opt, dataset)

    inputs = torch.cat([input_0, input_1], dim=0)

    if dataset == 'test':
        vertices = torch.cat([vertices_0, vertices_1], dim=0)
    else:
        vertices = torch.matmul(opt["evecs"], inputs)

    # compute spatial interp
    inputs_spatial_interp = (1-a) * inputs[0].unsqueeze(0) + a * inputs[1].unsqueeze(0)
    inputs_spatial_interp = torch.matmul(opt['evecs'], inputs_spatial_interp)

    # compute latent interpolation
    latents = opt["model"].enc(inputs)
    sum_latents = ((1-a) * latents[0] + a * latents[1]).unsqueeze(0)
    rec_spectral = opt["model"].dec(sum_latents)

    rec_spatial = torch.matmul(opt['evecs'], rec_spectral)

    # Display
    register_surface(vertices[0], "Input 0", opt['TRIV'], 0, transparency=0.25)
    register_surface(vertices[1], "Input 1", opt['TRIV'], 0, transparency=0.25)
    register_surface(rec_spatial[0], "Latent", opt['TRIV'], 1)
    register_surface(inputs_spatial_interp[0], "Spatial", opt['TRIV'], 5, transparency=0.75)

    print()

    return inputs


def interp_a(opt, all_coeffs, a):
    # compute spatial interpolation
    coeffs_interp_i = (1 - a) * all_coeffs[0].unsqueeze(0) + a * all_coeffs[1].unsqueeze(0)
    vertices_interp_i = torch.matmul(opt['evecs'], coeffs_interp_i)

    # compute latent interpolation
    all_latents = opt["model"].enc(all_coeffs)
    sum_latents = ((1 - a) * all_latents[0] + a * all_latents[1]).unsqueeze(0)
    rec_spectral = opt["model"].dec(sum_latents)

    rec_spatial = torch.matmul(opt['evecs'], rec_spectral)

    # Display
    register_surface(rec_spatial[0], "Latent", opt['TRIV'], 1)
    register_surface(vertices_interp_i[0], "Spatial", opt['TRIV'], 5, transparency=0.75)


def get_opts(job_id):
    # Create options for model and Polyscope
    trainer, opt = load_trainer(job_id)

    opt_infos = "\nModel informations: \n\n"
    for key, value in opt.items():
        if key not in ['losses', "TRIV", "TRIV", "evecs", "dataloader_train", "dataloader_test"]:
            # print(key, ' : ', value)
            opt_infos += str(key) + ": " + str(value) + "\n"

    opt_ui = {}

    opt_ui["job_id"] = job_id
    opt_ui["dict_list"] = os.listdir("checkpoints/")

    opt_ui["wanted_index"] = 1
    opt_ui["train_dataset"] = False
    opt_ui["test_dataset"] = True

    opt_ui["start"] = True

    opt_ui["index_couple_0"] = 10
    opt_ui["index_couple_1"] = 871

    return opt_ui, opt, opt_infos


def callback():
    global opt_ui, opt, opt_infos

    psim.PushItemWidth(200)

    psim.SetNextItemOpen(True)
    if (psim.TreeNode("Choose model")):
        psim.PushItemWidth(200)
        changed = psim.BeginCombo("Pick a model", opt_ui["job_id"])
        if changed:
            for val in opt_ui["dict_list"]:
                _, selected = psim.Selectable(val, opt_ui["job_id"] == val)
                if selected:
                    opt_ui["job_id"] = val
            psim.EndCombo()
        psim.PopItemWidth()

        if psim.Button("Load model"):
            opt_ui, opt, opt_infos = get_opts(opt_ui["job_id"])

        if psim.TreeNode("Infos"):
            psim.TextUnformatted(opt_infos)

            psim.TreePop()

        psim.TreePop()

    if opt_ui["train_dataset"]:
        dataset = 'train'
        max = opt['nb_train']-1
    else:
        dataset = 'test'
        max = opt['nb_evals']-1

    # Choose train or test dataset
    changed, opt_ui["train_dataset"] = psim.Checkbox("Train dataset", opt_ui["train_dataset"])

    if changed:
        opt_ui["test_dataset"] = not opt_ui["train_dataset"]

    psim.SameLine()

    changed, opt_ui["test_dataset"] = psim.Checkbox("Test dataset", opt_ui["test_dataset"])

    if changed:
        opt_ui["train_dataset"] = not opt_ui["test_dataset"]

    # TreeNode Reconstruction
    psim.SetNextItemOpen(True)
    if psim.TreeNode("Reconstruction"):
        psim.TextUnformatted("Train dataset size: {}".format(opt['nb_train']))
        psim.TextUnformatted("Test dataset size: {}".format(opt['nb_evals']))

        # Choose index for reconstruction
        _, opt_ui["wanted_index"] = psim.InputInt("Wanted mesh index", opt_ui["wanted_index"])

        if opt_ui["wanted_index"] > max:
            opt_ui["wanted_index"] = max

        if psim.Button("Random index"):
            opt_ui["wanted_index"] = random.randint(0, max)
        
        # Load reconstruction
        if psim.Button("Load reconstruction") or opt_ui["start"]:
            ps.remove_all_structures()
            opt_ui["start"] = False
            _ = reconstruction(opt_ui["wanted_index"], opt, dataset)

        psim.TreePop()

    # TreeNode Latent interpolation
    psim.SetNextItemOpen(True)
    if psim.TreeNode("Latent interpolation"):
        # Choose indices
        _, opt_ui["index_couple_0"] = psim.InputInt("Index 0", opt_ui["index_couple_0"])
        _, opt_ui["index_couple_1"] = psim.InputInt("Index 1", opt_ui["index_couple_1"])

        if opt_ui["index_couple_0"] > max:
            opt_ui["index_couple_0"] = max

        if opt_ui["index_couple_1"] > max:
            opt_ui["index_couple_1"] = max

        if psim.Button("Random indices"):
            opt_ui["index_couple_0"] = random.randint(0, max)
            opt_ui["index_couple_1"] = random.randint(0, max)

        # Some interesting indices of meshes to visualise interpolation
        interesting_indices = [[434, 9863], [5635, 9891], [1307, 3814], [5977, 10415]]
        if psim.TreeNode("Interesting indices"):
            if psim.TreeNode("AMASS"):
                text = [str(interesting_indices[i][0]) + "-" + str(interesting_indices[i][1]) + "\n"
                        for i in range(len(interesting_indices))]
                psim.TextUnformatted(''.join(text))

            psim.TreePop()

        psim.TreePop()

        if psim.Button("Load interp"):
            ps.remove_all_structures()

            opt_ui["all_coeffs"] = interp([opt_ui["index_couple_0"], opt_ui["index_couple_1"]], opt, dataset)
            opt_ui["a"] = 0.5

        if "all_coeffs" in opt_ui:
            changed_a, opt_ui["a"] = psim.SliderFloat("Interp a", opt_ui["a"], v_min=0, v_max=1)

            if changed_a:
                interp_a(opt, opt_ui["all_coeffs"], opt_ui["a"])

        psim.TreePop()


if __name__ == "__main__":
    with torch.no_grad():
        job_id = "SAE-LP-4096-16"

        opt_ui, opt, opt_infos = get_opts(job_id)

        ps.init()
        ps.set_up_dir("z_up")
        ps.set_ground_plane_mode("shadow_only")
        ps.set_ground_plane_height_factor(0)  # adjust the plane height

        ps.set_user_callback(callback)
        ps.show()
