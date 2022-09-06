import mph
import numpy as np
from jpype import JBoolean, JInt, JString, JArray, JDouble
from tqdm import tqdm
import time



def generate(client, k_matcustom0, k_matcustom1, matrix_mat):
    global start_time
    start_time = time.time()

    model = client.create('Model').java

    model.modelPath(".");

    model.label("top_opt_clean_version4.mph");

    model.param().set("r1", "50[cm]");
    model.param().set("a1", "20[cm]");
    model.param().set("a2", "10[mm]");
    model.param().set("p", "1[cm]");
    model.param().set("Q", "10000");

    model.component().create("comp1", JBoolean(True));

    model.component("comp1").geom().create("geom1", 2);

    model.component("comp1").mesh().create("mesh1");

    num = 0
    temp_time = time.time()

    for i in tqdm(np.arange(-0.095, 0.096, 0.01), desc="Generating boxes"):
        for j in np.arange(0.095, -0.096, -0.01):
            # print(np.around(i,3),",",np.around(j,3))
            num += 1
            model.component("comp1").geom("geom1").create("sq" + str(num), "Square");
            model.component("comp1").geom("geom1").feature("sq" + str(num)).set("pos", [np.around(i, 3).item(),
                                                                                        np.around(j, 3).item()]);
            model.component("comp1").geom("geom1").feature("sq" + str(num)).set("base", "center");
            model.component("comp1").geom("geom1").feature("sq" + str(num)).set("size", "a2");
    print('Completed {} boxes  costs: {:2.2f} sec(s), total used: {:2.2f} sec(s)'.format(str(num), time.time() - start_time,
                                                             time.time() - temp_time))




    # print("Completed", str(num), " boxes", "Time cost")
    model.component("comp1").geom("geom1").run();

    model.component("comp1").geom("geom1").create("c_bound", "Circle");
    model.component("comp1").geom("geom1").feature("c_bound").set("r", "r1");
    model.component("comp1").geom("geom1").create("sq_mat", "Square");
    model.component("comp1").geom("geom1").feature("sq_mat").set("base", "center");
    model.component("comp1").geom("geom1").feature("sq_mat").set("size", "a1");
    model.component("comp1").geom("geom1").create("cfield", "Circle");
    model.component("comp1").geom("geom1").feature("cfield").set("r", "a1");

    model.component("comp1").geom("geom1").create("c1", "Circle");
    model.component("comp1").geom("geom1").feature("c1").label("heatsource0");
    model.component("comp1").geom("geom1").feature("c1").set("r", "p");
    model.component("comp1").geom("geom1").create("c2", "Circle");
    model.component("comp1").geom("geom1").feature("c2").label("heatsource1");
    model.component("comp1").geom("geom1").feature("c2").set("pos", [0.08, 0.08]);
    model.component("comp1").geom("geom1").feature("c2").set("r", "p");
    model.component("comp1").geom("geom1").create("c3", "Circle");
    model.component("comp1").geom("geom1").feature("c3").label("heatsource2");
    model.component("comp1").geom("geom1").feature("c3").set("pos", [-0.08, 0.08]);
    model.component("comp1").geom("geom1").feature("c3").set("r", "p");
    model.component("comp1").geom("geom1").create("c4", "Circle");
    model.component("comp1").geom("geom1").feature("c4").label("heatsource3");
    model.component("comp1").geom("geom1").feature("c4").set("pos", [0.08, -0.08]);
    model.component("comp1").geom("geom1").feature("c4").set("r", "p");
    model.component("comp1").geom("geom1").create("c5", "Circle");
    model.component("comp1").geom("geom1").feature("c5").label("heatsource4");
    model.component("comp1").geom("geom1").feature("c5").set("pos", [-0.08, -0.08]);
    model.component("comp1").geom("geom1").feature("c5").set("r", "p");

    num = 0
    temp_time = time.time()
    for i in tqdm(np.arange(-0.1, 0.1, 0.01), desc="Generating selections"):
        for j in np.arange(0.1, -0.1, -0.01):
            num += 1
            xmin = np.around(i, 3).item()
            xmax = np.around(i + 0.01, 3).item()
            ymin = np.around(j - 0.01, 3).item()
            ymax = np.around(j, 3).item()
            model.component("comp1").selection().create("box" + str(num), "Box");
            model.component("comp1").selection("box" + str(num)).set("xmin", str(xmin));
            model.component("comp1").selection("box" + str(num)).set("xmax", str(xmax));
            model.component("comp1").selection("box" + str(num)).set("ymin", str(ymin));
            model.component("comp1").selection("box" + str(num)).set("ymax", str(ymax));
            model.component("comp1").selection("box" + str(num)).set("condition", "allvertices");
    # print("Completed", str(num), " selections")
    print('Completed {} selections  costs: {:2.2f} sec(s), total used: {:2.2f} sec(s)'.format(str(num), time.time() - start_time,
                                                             time.time() - temp_time))

    model.component("comp1").selection().create("disk0", "Disk");
    # model.component("comp1").selection("disk0").set("posx", 0.0);
    # model.component("comp1").selection("disk0").set("posy", 0.0);
    model.component("comp1").selection("disk0").set("r", "p");
    model.component("comp1").selection("disk0").set("condition", "allvertices");

    model.component("comp1").selection().create("disk1", "Disk");
    model.component("comp1").selection("disk1").set("posx", "0.08");
    model.component("comp1").selection("disk1").set("posy", "0.08");
    model.component("comp1").selection("disk1").set("r", "p");
    model.component("comp1").selection("disk1").set("condition", "allvertices");

    model.component("comp1").selection().create("disk2", "Disk");
    model.component("comp1").selection("disk2").set("posx", "-0.08");
    model.component("comp1").selection("disk2").set("posy", "0.08");
    model.component("comp1").selection("disk2").set("r", "p");
    model.component("comp1").selection("disk2").set("condition", "allvertices");

    model.component("comp1").selection().create("disk3", "Disk");
    model.component("comp1").selection("disk3").set("posx", "0.08");
    model.component("comp1").selection("disk3").set("posy", "-0.08");
    model.component("comp1").selection("disk3").set("r", "p");
    model.component("comp1").selection("disk3").set("condition", "allvertices");

    model.component("comp1").selection().create("disk4", "Disk");
    model.component("comp1").selection("disk4").set("posx", "-0.08");
    model.component("comp1").selection("disk4").set("posy", "-0.08");
    model.component("comp1").selection("disk4").set("r", "p");
    model.component("comp1").selection("disk4").set("condition", "allvertices");

    model.component("comp1").selection().create("boxmat", "Box");
    model.component("comp1").selection("boxmat").set("xmin", -0.1);
    model.component("comp1").selection("boxmat").set("xmax", 0.1);
    model.component("comp1").selection("boxmat").set("ymin", -0.1);
    model.component("comp1").selection("boxmat").set("ymax", 0.1);
    model.component("comp1").selection("boxmat").set("condition", "allvertices");

    model.component("comp1").selection().create("diskmat", "Disk");
    # model.component("comp1").selection("diskmat").set("posx", 0);
    # model.component("comp1").selection("diskmat").set("posy", 0);
    model.component("comp1").selection("diskmat").set("r", "a1");
    model.component("comp1").selection("diskmat").set("condition", "allvertices");

    model.component("comp1").selection().create("difprobe", "Difference");
    model.component("comp1").selection("difprobe").set("add", "diskmat")
    model.component("comp1").selection("difprobe").set("subtract", "boxmat")

    model.component("comp1").selection().create("diskbnd", "Disk");
    model.component("comp1").selection("diskbnd").set("entitydim", JInt(1));
    model.component("comp1").selection("diskbnd").set("r", "r1");
    model.component("comp1").selection("diskbnd").set("rin", "r1-0.001");
    model.component("comp1").selection("diskbnd").set("condition", "somevertex");

    # end of selection

    model.component("comp1").material().create("mat1", "Common");
    model.component("comp1").material("mat1").propertyGroup().create("Enu", "Young's modulus and Poisson's ratio");
    model.component("comp1").material("mat1").label("FR4 (Circuit Board)");
    model.component("comp1").material("mat1").set("family", "pcbgreen");
    model.component("comp1").material("mat1").propertyGroup("def").set("relpermeability",
                                                                       ["1", "0", "0", "0", "1", "0", "0", "0", "1"]);
    model.component("comp1").material("mat1").propertyGroup("def").set("electricconductivity",
                                                                       ["0.004[S/m]", "0", "0", "0", "0.004[S/m]", "0",
                                                                        "0", "0", "0.004[S/m]"]);
    model.component("comp1").material("mat1").propertyGroup("def").set("relpermittivity",
                                                                       ["4.5", "0", "0", "0", "4.5", "0", "0", "0",
                                                                        "4.5"]);
    model.component("comp1").material("mat1").propertyGroup("def").set("thermalexpansioncoefficient",
                                                                       ["18e-6[1/K]", "0", "0", "0", "18e-6[1/K]", "0",
                                                                        "0", "0", "18e-6[1/K]"]);
    model.component("comp1").material("mat1").propertyGroup("def").set("heatcapacity", "1369[J/(kg*K)]");
    model.component("comp1").material("mat1").propertyGroup("def").set("density", "1900[kg/m^3]");
    model.component("comp1").material("mat1").propertyGroup("def").set("thermalconductivity",
                                                                       ["0.3[W/(m*K)]", "0", "0", "0", "0.3[W/(m*K)]",
                                                                        "0", "0", "0", "0.3[W/(m*K)]"]);
    model.component("comp1").material("mat1").propertyGroup("Enu").set("youngsmodulus", "22e9[Pa]");
    model.component("comp1").material("mat1").propertyGroup("Enu").set("poissonsratio", "0.15");

    model.component("comp1").physics().create("ht", "HeatTransfer", "geom1");
    model.component("comp1").physics().create("ht2", "HeatTransfer", "geom1");

    model.component("comp1").physics("ht").create("hs1", "HeatSource", JInt(2));
    model.component("comp1").physics("ht").feature("hs1").selection().set().named("disk1");
    model.component("comp1").physics("ht").feature("hs1").set("Q0", "Q/4");
    model.component("comp1").physics("ht").create("hs2", "HeatSource", JInt(2));
    model.component("comp1").physics("ht").feature("hs2").selection().set().named("disk2");
    model.component("comp1").physics("ht").feature("hs2").set("Q0", "Q/4");
    model.component("comp1").physics("ht").create("hs3", "HeatSource", JInt(2));
    model.component("comp1").physics("ht").feature("hs3").selection().set().named("disk3");
    model.component("comp1").physics("ht").feature("hs3").set("Q0", "Q/4");
    model.component("comp1").physics("ht").create("hs4", "HeatSource", JInt(2));
    model.component("comp1").physics("ht").feature("hs4").selection().set().named("disk4");
    model.component("comp1").physics("ht").feature("hs4").set("Q0", "Q/4");

    model.component("comp1").physics("ht2").create("hs0", "HeatSource", JInt(2));
    model.component("comp1").physics("ht2").feature("hs0").selection().named("disk0");
    model.component("comp1").physics("ht2").feature("hs0").set("Q0", "Q");

    model.component("comp1").physics("ht").create("temp1", "TemperatureBoundary", JInt(1));
    model.component("comp1").physics("ht").feature("temp1").selection().named("diskbnd");

    model.component("comp1").physics("ht2").create("temp1", "TemperatureBoundary", JInt(1));
    model.component("comp1").physics("ht2").feature("temp1").selection().named("diskbnd");

    # num = 0;
    # temp_time = time.time()
    # for i in tqdm(range(0, 200), desc="Generating solids"):
    #     for j in range(0, 200):
    #         num += 1
    #         temp = matrix_mat[i, j].item()*95.9+0.3
    #         model.component("comp1").physics("ht2").create("solidcustom" + str(num), "SolidHeatTransferModel", JInt(2));
    #
    #         model.component("comp1").physics("ht2").feature("solidcustom" + str(num)).set("k_mat", "userdef");
    #         model.component("comp1").physics("ht2").feature("solidcustom" + str(num)).set("k",
    #                                                                             [temp, 0, 0, 0, temp, 0, 0, 0, temp]);
    #         model.component("comp1").physics("ht2").feature("solidcustom1").selection().named("box"+str(num));
    # # print("Completed " + str(num) + " solids")
    # print('Completed {} selections  costs: {:2.2f} sec(s), total used: {:2.2f} sec(s)'.format(str(num), time.time() - start_time,
    #                                                          time.time() - temp_time))

    model.component("comp1").physics("ht2").create("solidcustom1", "SolidHeatTransferModel", JInt(2));
    model.component("comp1").physics("ht2").feature("solidcustom1").set("k",  [k_matcustom0, 0, 0, 0, k_matcustom0, 0, 0, 0, k_matcustom0]);

    model.component("comp1").physics("ht2").create("solidcustom2", "SolidHeatTransferModel", JInt(2));
    model.component("comp1").physics("ht2").feature("solidcustom2").set("k",
                                                                                   [k_matcustom1, 0, 0, 0, k_matcustom1,
                                                                                    0, 0, 0, k_matcustom1]);

    num = 0;
    Sel0 = [];
    Sel1 = [];
    for i in range(0, 20):
        for j in range(0, 20):
            num += 1
            if (matrix_mat[i, j].item() == 0):

                Sel0.extend(model.component("comp1").selection("box" + str(num)).entities())
                # model.component("comp1").physics("ht2").feature("solidcustom1").selection().add().named(strSel0);
                # strSel0 += "box"+str(num)+","
            else:
                # strSel1 += "box"+str(num)+","
                Sel1.extend(model.component("comp1").selection("box" + str(num)).entities())
                # model.component("comp1").physics("ht2").feature("solidcustom2").selection().add().named(strSel1);
    print("Sel0:")
    print(Sel0)
    print("Sel1:")
    print(Sel1)
    model.component("comp1").physics("ht2").feature("solidcustom1").selection().set(Sel0);
    model.component("comp1").physics("ht2").feature("solidcustom2").selection().set(Sel1);
    print("Completed " + str(num) + " solids")

    model.component("comp1").probe().create("dom1", "Domain");
    model.component("comp1").probe("dom1").selection().named("difprobe");
    model.component("comp1").probe("dom1").set("expr", "abs(T-T2)");
    model.component("comp1").probe("dom1").set("descr", "abs(T-T2)");
    model.result().table().create("tbl1", "Table")

    # model.result("tbl1").set("graphics", "off")
    model.component("comp1").probe("dom1").set("table", "tbl1");

    print("Starting create std1")
    temp_time = time.time()
    model.study().create("std1");
    model.study("std1").create("stat", "Stationary");
    model.sol().create("sol1");
    model.sol("sol1").study("std1");
    model.sol("sol1").attach("std1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").create("s1", "Stationary");
    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("s1").create("d1", "Direct");
    model.sol("sol1").feature("s1").create("i1", "Iterative");
    model.sol("sol1").feature("s1").feature("i1").create("mg1", "Multigrid");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").create("so1", "SOR");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").create("so1", "SOR");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("cs").create("d1", "Direct");
    model.sol("sol1").feature("s1").feature().remove("fcDef");
    print('Completed {} create std1  costs: {:2.2f} sec(s), total used: {:2.2f} sec(s)'.format(str(num),
                                                                                              time.time() - start_time,
                                                                                              time.time() - temp_time))
    print("Starting create plot")
    temp_time = time.time()

    model.result().create("pg1", "PlotGroup2D");
    model.result().create("pg2", "PlotGroup2D");
    model.result().create("pg3", "PlotGroup2D");
    model.result().create("pg4", "PlotGroup2D");
    model.result().create("pg5", "PlotGroup2D");
    model.result("pg1").create("surf1", "Surface");
    model.result("pg2").create("con1", "Contour");
    model.result("pg3").create("surf1", "Surface");
    model.result("pg3").feature("surf1").set("expr", "T2");
    model.result("pg4").create("con1", "Contour");
    model.result("pg4").feature("con1").set("expr", "T2");
    model.result("pg5").create("surf1", "Surface");
    model.result("pg5").create("con1", "Contour");
    model.result("pg5").feature("surf1").set("expr", "T-T2");
    model.result("pg5").feature("con1").set("expr", "T-T2");
    print('Completed {} create plot  costs: {:2.2f} sec(s), total used: {:2.2f} sec(s)'.format(str(num),
                                                                                              time.time() - start_time,

                                                                                              time.time() - temp_time))

    model.component("comp1").probe("dom1").genResult("sol1");

    print("Starting create sol1")
    model.sol("sol1").attach("std1");
    model.sol("sol1").feature("st1").label("Compile Equations: Stationary");
    model.sol("sol1").feature("v1").label("Dependent Variables 1.1");
    model.sol("sol1").feature("s1").label("Stationary Solver 1.1");
    model.sol("sol1").feature("s1").feature("dDef").label("Direct 2");
    model.sol("sol1").feature("s1").feature("aDef").label("Advanced 1");
    model.sol("sol1").feature("s1").feature("fc1").label("Fully Coupled 1.1");
    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("s1").feature("fc1").set("initstep", 0.01);
    model.sol("sol1").feature("s1").feature("fc1").set("minstep", 1.0E-6);
    model.sol("sol1").feature("s1").feature("fc1").set("maxiter", JInt(50));
    model.sol("sol1").feature("s1").feature("fc1").set("termonres", JBoolean(False));
    model.sol("sol1").feature("s1").feature("d1").label("Direct, temperature (ht) (merged)");
    model.sol("sol1").feature("s1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("s1").feature("d1").set("pivotperturb", 1.0E-13);
    model.sol("sol1").feature("s1").feature("i1").label("AMG, temperature (ht)");
    model.sol("sol1").feature("s1").feature("i1").set("nlinnormuse", JBoolean(True));
    model.sol("sol1").feature("s1").feature("i1").set("rhob", JInt(20));
    model.sol("sol1").feature("s1").feature("i1").feature("ilDef").label("Incomplete LU 1");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").label("Multigrid 1.1");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").set("prefun", "saamg");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").set("maxcoarsedof", JInt(50000));
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").set("saamgcompwise", JBoolean(True));
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").set("usesmooth", JBoolean(False));
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").label("Presmoother 1");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("soDef").label("SOR 2");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("pr").feature("so1").label("SOR 1.1");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").label("Postsmoother 1");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("soDef").label("SOR 2");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("po").feature("so1").label("SOR 1.1");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("cs").label("Coarse Solver 1");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("cs").feature("dDef").label("Direct 2");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("cs").feature("d1").label("Direct 1.1");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("cs").feature("d1").set("linsolver",
                                                                                                 "pardiso");
    model.sol("sol1").feature("s1").feature("i1").feature("mg1").feature("cs").feature("d1").set("pivotperturb",
                                                                                                 1.0E-13);
    print('Completed {} create sol1 costs: {:2.2f} sec(s), total used: {:2.2f} sec(s)'.format(str(num),
                                                                                               time.time() - start_time,

                                                                                               time.time() - temp_time))

    # model.result().autoplot(JBoolean(False))

    print("Model computing...")
    temp_time = time.time()
    model.sol("sol1").runAll();
    print('Completed {} computing  costs: {:2.2f} sec(s), total used: {:2.2f} sec(s)'.format(str(num), time.time() - start_time,
                                                             time.time() - temp_time))
    print("Completed!")

    model.result("pg1").label("Temperature (ht)");
    model.result("pg1").feature("surf1").label("Surface");
    model.result("pg1").feature("surf1").set("colortable", "Rainbow");
    model.result("pg1").feature("surf1").set("resolution", "normal");
    model.result("pg2").label("Isothermal Contours (ht)");
    model.result("pg2").feature("con1").label("Contour");
    model.result("pg2").feature("con1").set("levelrounding", JBoolean(False));
    model.result("pg2").feature("con1").set("smooth", "internal");
    model.result("pg2").feature("con1").set("resolution", "normal");
    model.result("pg3").label("Temperature (ht2)");
    model.result("pg3").feature("surf1").label("Surface");
    model.result("pg3").feature("surf1").set("colortable", "Rainbow");
    model.result("pg3").feature("surf1").set("resolution", "normal");
    model.result("pg4").label("Isothermal Contours (ht2)");
    model.result("pg4").feature("con1").label("Contour");
    model.result("pg4").feature("con1").set("levelrounding", JBoolean(False));
    model.result("pg4").feature("con1").set("smooth", "internal");
    model.result("pg4").feature("con1").set("resolution", "normal");
    model.result("pg5").label("Dif");
    model.result("pg5").feature("surf1").active(JBoolean(False));
    model.result("pg5").feature("surf1").set("resolution", "normal");
    model.result("pg5").feature("con1").set("resolution", "normal");

    probe_t = model.result().table("tbl1").getRealRow(0)[0] * 100
    # probe_t = 0


    print(matrix_mat)
    return model, probe_t


def opt(model, matrix_mat):
    model = model.java
    num = 0;
    Sel0 = [];
    Sel1 = [];
    for i in range(0, 20):
        for j in range(0, 20):
            num += 1
            if (matrix_mat[i, j].item() == 0):
                Sel0.extend(model.component("comp1").selection("box" + str(num)).entities())
            else:
               Sel1.extend(model.component("comp1").selection("box" + str(num)).entities())


    model.component("comp1").physics("ht2").feature("solidcustom1").selection().set(Sel0);
    model.component("comp1").physics("ht2").feature("solidcustom2").selection().set(Sel1);
    model.sol("sol1").runAll();

    probe_t = model.result().table("tbl1").getRealRow(0)[0] * 100
    print("!")
    return probe_t

def pp():
    import os
    client = mph.start()
    for i in ["array","array25x","array25x/dom1_difprobe"]:
    # for i in ["array25x/dom1_difprobe"]:

            # if (i==0)&(0<j<7):
            #     print("skip {}{}".format(i,j))
            #     continue
            # if (i==1)&(0<j<3):
            #     print("skip {}{}".format(i,j))
            #     continue



            os.chdir("./{}".format(i))
            print("Load {}".format(i))
            model = client.load("array_final.mph").java
            print("Soving...")
            model.sol("sol1").runAll();
            model.result("pg1").label("01");
            model.result("pg1").set("edges", JBoolean(False));
            model.result("pg1").feature("surf1").set("colortable", "HeatCamera");
            model.result("pg1").feature("surf1").set("resolution", "normal");
            model.result("pg3").label("02");
            model.result("pg3").set("edges", JBoolean(False));
            model.result("pg3").feature("surf1").set("colortable", "HeatCamera");
            model.result("pg3").feature("surf1").set("resolution", "normal");
            model.result("pg5").label("03");
            model.result("pg5").set("edges", JBoolean(False));
            model.result("pg5").feature("surf1").set("colortable", "HeatCamera");
            model.result("pg5").feature("surf1").set("expr", "abs(T-T2)");
            model.result("pg5").feature().remove("con1");

            model.result("pg1").selection().named("diskmat");
            model.result("pg3").selection().named("diskmat");
            model.result("pg5").selection().named("diskmat");

            model.result("pg1").run();
            model.result("pg3").run();
            model.result("pg5").run();

            model.result().export().create("img1", "Image");
            model.result().export().create("img2", "Image");
            model.result().export().create("img3", "Image");
            model.result().export("img1").set("sourceobject", "pg1");
            model.result().export("img1").set("lockratio", JBoolean(False));
            model.result().export("img1").set("pngfilename", "01");
            model.result().export("img1").set("size", "manualweb");
            model.result().export("img1").set("unit", "px");
            model.result().export("img1").set("height", "1500");
            model.result().export("img1").set("width", "2000");
            model.result().export("img1").set("lockratio", "off");
            model.result().export("img1").set("resolution", "96");
            model.result().export("img1").set("antialias", "on");
            model.result().export("img1").set("zoomextents", "on");
            model.result().export("img1").set("fontsize", "25");
            model.result().export("img1").set("colortheme", "globaltheme");
            model.result().export("img1").set("customcolor", [1.0, 1.0, 1.0]);
            model.result().export("img1").set("background", "transparent");
            model.result().export("img1").set("gltfincludelines", "on");
            model.result().export("img1").set("title1d", "on");
            model.result().export("img1").set("legend1d", "on");
            model.result().export("img1").set("logo1d", "on");
            model.result().export("img1").set("options1d", "on");
            model.result().export("img1").set("title2d", "off");
            model.result().export("img1").set("legend2d", "on");
            model.result().export("img1").set("logo2d", "off");
            model.result().export("img1").set("options2d", "on");
            model.result().export("img1").set("title3d", "on");
            model.result().export("img1").set("legend3d", "on");
            model.result().export("img1").set("logo3d", "on");
            model.result().export("img1").set("options3d", "off");
            model.result().export("img1").set("axisorientation", "on");
            model.result().export("img1").set("grid", "on");
            model.result().export("img1").set("axes1d", "on");
            model.result().export("img1").set("axes2d", "off");
            model.result().export("img1").set("showgrid", "on");
            model.result().export("img1").set("target", "file");
            model.result().export("img1").set("qualitylevel", "92");
            model.result().export("img1").set("qualityactive", "off");
            model.result().export("img1").set("imagetype", "png");
            model.result().export("img1").set("lockview", "off");
            model.result().export("img2").set("sourceobject", "pg3");
            model.result().export("img2").set("lockratio", JBoolean(False));
            model.result().export("img2").set("pngfilename", "2");
            model.result().export("img2").set("size", "manualweb");
            model.result().export("img2").set("unit", "px");
            model.result().export("img2").set("height", "1500");
            model.result().export("img2").set("width", "2000");
            model.result().export("img2").set("lockratio", "off");
            model.result().export("img2").set("resolution", "96");
            model.result().export("img2").set("antialias", "on");
            model.result().export("img2").set("zoomextents", "on");
            model.result().export("img2").set("fontsize", "25");
            model.result().export("img2").set("colortheme", "globaltheme");
            model.result().export("img2").set("customcolor", [1.0, 1.0, 1.0]);
            model.result().export("img2").set("background", "transparent");
            model.result().export("img2").set("gltfincludelines", "on");
            model.result().export("img2").set("title1d", "on");
            model.result().export("img2").set("legend1d", "on");
            model.result().export("img2").set("logo1d", "on");
            model.result().export("img2").set("options1d", "on");
            model.result().export("img2").set("title2d", "off");
            model.result().export("img2").set("legend2d", "on");
            model.result().export("img2").set("logo2d", "off");
            model.result().export("img2").set("options2d", "on");
            model.result().export("img2").set("title3d", "on");
            model.result().export("img2").set("legend3d", "on");
            model.result().export("img2").set("logo3d", "on");
            model.result().export("img2").set("options3d", "off");
            model.result().export("img2").set("axisorientation", "on");
            model.result().export("img2").set("grid", "on");
            model.result().export("img2").set("axes1d", "on");
            model.result().export("img2").set("axes2d", "off");
            model.result().export("img2").set("showgrid", "on");
            model.result().export("img2").set("target", "file");
            model.result().export("img2").set("qualitylevel", "92");
            model.result().export("img2").set("qualityactive", "off");
            model.result().export("img2").set("imagetype", "png");
            model.result().export("img2").set("lockview", "off");
            model.result().export("img3").set("sourceobject", "pg5");
            model.result().export("img3").set("lockratio", JBoolean(False));
            model.result().export("img3").set("pngfilename", "3");
            model.result().export("img3").set("size", "manualweb");
            model.result().export("img3").set("unit", "px");
            model.result().export("img3").set("height", "1500");
            model.result().export("img3").set("width", "2000");
            model.result().export("img3").set("lockratio", "off");
            model.result().export("img3").set("resolution", "96");
            model.result().export("img3").set("antialias", "on");
            model.result().export("img3").set("zoomextents", "on");
            model.result().export("img3").set("fontsize", "25");
            model.result().export("img3").set("colortheme", "globaltheme");
            model.result().export("img3").set("customcolor", [1.0, 1.0, 1.0]);
            model.result().export("img3").set("background", "transparent");
            model.result().export("img3").set("gltfincludelines", "on");
            model.result().export("img3").set("title1d", "on");
            model.result().export("img3").set("legend1d", "on");
            model.result().export("img3").set("logo1d", "on");
            model.result().export("img3").set("options1d", "on");
            model.result().export("img3").set("title2d", "off");
            model.result().export("img3").set("legend2d", "on");
            model.result().export("img3").set("logo2d", "off");
            model.result().export("img3").set("options2d", "on");
            model.result().export("img3").set("title3d", "on");
            model.result().export("img3").set("legend3d", "on");
            model.result().export("img3").set("logo3d", "on");
            model.result().export("img3").set("options3d", "off");
            model.result().export("img3").set("axisorientation", "on");
            model.result().export("img3").set("grid", "on");
            model.result().export("img3").set("axes1d", "on");
            model.result().export("img3").set("axes2d", "off");
            model.result().export("img3").set("showgrid", "on");
            model.result().export("img3").set("target", "file");
            model.result().export("img3").set("qualitylevel", "92");
            model.result().export("img3").set("qualityactive", "off");
            model.result().export("img3").set("imagetype", "png");
            model.result().export("img3").set("lockview", "off");

            model.result().export("img1").set("zoomextents", JBoolean(True));
            model.result().export("img2").set("zoomextents", JBoolean(True));
            model.result().export("img3").set("zoomextents", JBoolean(True));

            print("Ploting...")
            model.result().export("img1").run()
            model.result().export("img2").run()
            model.result().export("img3").run()
            print("Complete {}".format(i))
            client.clear()
            os.chdir("../")


