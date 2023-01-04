#include "BasicScene.h"
#include <Eigen/src/Core/Matrix.h>
#include <edges.h>
#include <memory>
#include <per_face_normals.h>
#include <read_triangle_mesh.h>
#include <utility>
#include <vector>
#include "GLFW/glfw3.h"
#include "Mesh.h"
#include "PickVisitor.h"
#include "Renderer.h"
#include "ObjLoader.h"
#include "IglMeshLoader.h"

#include "igl/per_vertex_normals.h"
#include "igl/per_face_normals.h"
#include "igl/unproject_onto_mesh.h"
#include "igl/edge_flaps.h"
#include "igl/loop.h"
#include "igl/upsample.h"
#include "igl/AABB.h"
#include "igl/parallel_for.h"
#include "igl/shortest_edge_and_midpoint.h"
#include "igl/circulation.h"
#include "igl/edge_midpoints.h"
#include "igl/collapse_edge.h"
#include "igl/edge_collapse_is_valid.h"
#include "igl/write_triangle_mesh.h"

// #include "AutoMorphingModel.h"

using namespace cg3d;

void BasicScene::Init(float fov, int width, int height, float near, float far)
{
    camera = Camera::Create( "camera", fov, float(width) / height, near, far);
    
    AddChild(root = Movable::Create("root")); // a common (invisible) parent object for all the shapes
    auto daylight{std::make_shared<Material>("daylight", "shaders/cubemapShader")}; 
    daylight->AddTexture(0, "textures/cubemaps/Daylight Box_", 3);
    auto background{Model::Create("background", Mesh::Cube(), daylight)};
    AddChild(background);
    background->Scale(120, Axis::XYZ);
    background->SetPickable(false);
    background->SetStatic();

 
    auto program = std::make_shared<Program>("shaders/phongShader");
    auto program1 = std::make_shared<Program>("shaders/pickingShader");
    
    auto material{ std::make_shared<Material>("material", program)}; // empty material
    auto material1{ std::make_shared<Material>("material", program1)}; // empty material
//    SetNamedObject(cube, Model::Create, Mesh::Cube(), material, shared_from_this());
 
    material->AddTexture(0, "textures/box0.bmp", 2);
    auto sphereMesh{IglLoader::MeshFromFiles("sphere_igl", "data/sphere.obj")};
    auto cylMesh{IglLoader::MeshFromFiles("cyl_igl","data/xcylinder.obj")};
    auto cubeMesh{IglLoader::MeshFromFiles("cube_igl","data/cube_old.obj")};
    sphere1 = Model::Create( "sphere",sphereMesh, material);    
    cube = Model::Create( "cube", cubeMesh, material);
    root->AddChild(sphere1);
    sphere1->Translate(5,Axis::X);
    float scaleFactor = 1; 
    //Axis
    Eigen::MatrixXd vertices(6,3);
    vertices << -1,0,0,1,0,0,0,-1,0,0,1,0,0,0,-1,0,0,1;
    Eigen::MatrixXi faces(3,2);
    faces << 0,1,2,3,4,5;
    Eigen::MatrixXd vertexNormals = Eigen::MatrixXd::Ones(6,3);
    Eigen::MatrixXd textureCoords = Eigen::MatrixXd::Ones(6,2);
    std::shared_ptr<Mesh> coordsys = std::make_shared<Mesh>("coordsys",vertices,faces,vertexNormals,textureCoords);
    axis.push_back(Model::Create("axis",coordsys,material1));
    axis[0]->mode = 1;   
    axis[0]->Scale(scaleFactor * 2,Axis::XYZ);
    axis[0]->lineWidth = 2;
    axis[0]->isPickable = false;
    //axis[0]->Translate(-0.8f * scaleFactor, Axis::X);
    cyls.push_back( Model::Create("cyl",cylMesh, material));
    cyls[0]->Scale(scaleFactor,Axis::X);
    cyls[0]->SetCenter(Eigen::Vector3f(-0.8f*scaleFactor,0,0));
    root->AddChild(cyls[0]);
    root->AddChild(axis[0]);
    cyls[0]->RotateByDegree(90*PI_DIV_180, Axis::Y);
    for(int i = 1;i < 4; i++)
    { 
        cyls.push_back( Model::Create("cyl", cylMesh, material));
        axis.push_back(Model::Create("axis", coordsys, material1));
        axis[i]->mode = 1;
        axis[i]->Scale(scaleFactor*2, Axis::XYZ);
        axis[i]->lineWidth = 2;
        axis[i]->isPickable = false;
        axis[i]->Translate(0.8f * scaleFactor, Axis::X);
        cyls[i]->Scale(scaleFactor,Axis::X);   
        cyls[i]->Translate(1.6f*scaleFactor,Axis::X);
        cyls[i]->SetCenter(Eigen::Vector3f(-0.8f*scaleFactor,0,0));
        cyls[i-1]->AddChild(cyls[i]);
        cyls[i-1]->AddChild(axis[i]);
    }
    cyls[0]->Translate({0.8f*scaleFactor,0,0});
    auto morphFunc = [](Model* model, cg3d::Visitor* visitor) {
      return model->meshIndex;
    };
    camera->Translate(22, Axis::Z);

}

void BasicScene::Update(const Program& program, const Eigen::Matrix4f& proj, const Eigen::Matrix4f& view, const Eigen::Matrix4f& model)
{
    Scene::Update(program, proj, view, model);
    program.SetUniform4f("lightColor", 0.8f, 0.3f, 0.0f, 0.5f);
    program.SetUniform4f("Kai", 1.0f, 0.3f, 0.6f, 1.0f);
    program.SetUniform4f("Kdi", 0.5f, 0.5f, 0.0f, 1.0f);
    program.SetUniform1f("specular_exponent", 5.0f);
    program.SetUniform4f("light_position", 0.0, 15.0f, 0.0, 1.0f);
//    cyl->Rotate(0.001f, Axis::Y);
   // cube->Rotate(0.1f, Axis::XYZ);
    if (isCCD)
        CCD();
    if (isFabrik)
        Fabrik();
   
}

void BasicScene::MouseCallback(Viewport* viewport, int x, int y, int button, int action, int mods, int buttonState[])
{
    // note: there's a (small) chance the button state here precedes the mouse press/release event

    if (action == GLFW_PRESS) { // default mouse button press behavior
        PickVisitor visitor;
        visitor.Init();
        renderer->RenderViewportAtPos(x, y, &visitor); // pick using fixed colors hack
        auto modelAndDepth = visitor.PickAtPos(x, renderer->GetWindowHeight() - y);
        renderer->RenderViewportAtPos(x, y); // draw again to avoid flickering
        pickedModel = modelAndDepth.first ? std::dynamic_pointer_cast<Model>(modelAndDepth.first->shared_from_this()) : nullptr;
        pickedModelDepth = modelAndDepth.second;
        camera->GetRotation().transpose();
        xAtPress = x;
        yAtPress = y;

        // if (pickedModel)
        //     debug("found ", pickedModel->isPickable ? "pickable" : "non-pickable", " model at pos ", x, ", ", y, ": ",
        //           pickedModel->name, ", depth: ", pickedModelDepth);
        // else
        //     debug("found nothing at pos ", x, ", ", y);
        if (isPickedLink() != -1)
            pickedIndex = isPickedLink();

        if (pickedModel && !pickedModel->isPickable)
            pickedModel = nullptr; // for non-pickable models we need only pickedModelDepth for mouse movement calculations later

        if (pickedModel)
            pickedToutAtPress = pickedModel->GetTout();
        else
            cameraToutAtPress = camera->GetTout();
    }
}

void BasicScene::ScrollCallback(Viewport* viewport, int x, int y, int xoffset, int yoffset, bool dragging, int buttonState[])
{
    // note: there's a (small) chance the button state here precedes the mouse press/release event
    auto system = camera->GetRotation().transpose();
    if (pickedModel) {
        if(isPickedLink()!=-1){
            cyls[0]->TranslateInSystem(system, {0, 0, -float(yoffset)});
            pickedToutAtPress = cyls[0]->GetTout();
        }
        else {
            pickedModel->TranslateInSystem(system, { 0, 0, -float(yoffset) });
            pickedToutAtPress = pickedModel->GetTout();
        }
    } else {
        camera->TranslateInSystem(system, {0, 0, -float(yoffset)});
        cameraToutAtPress = camera->GetTout();
    }
}

void BasicScene::CursorPosCallback(Viewport* viewport, int x, int y, bool dragging, int* buttonState)
{
    if (dragging) {
        auto system = camera->GetRotation().transpose() * GetRotation();
        auto moveCoeff = camera->CalcMoveCoeff(pickedModelDepth, viewport->width);
        auto angleCoeff = camera->CalcAngleCoeff(viewport->width);
        if (pickedModel) {
            //pickedModel->SetTout(pickedToutAtPress);
            if (isPickedLink()!=-1) {
                if (buttonState[GLFW_MOUSE_BUTTON_RIGHT] != GLFW_RELEASE)
                    cyls[0]->TranslateInSystem(system, { -float(xAtPress - x) / moveCoeff, float(yAtPress - y) / moveCoeff, 0 });
                if (buttonState[GLFW_MOUSE_BUTTON_MIDDLE] != GLFW_RELEASE)
                    cyls[0]->RotateInSystem(system, float(xAtPress - x) / angleCoeff, Axis::Z);
                if (buttonState[GLFW_MOUSE_BUTTON_LEFT] != GLFW_RELEASE) {
                    cyls[0]->RotateInSystem(system, float(xAtPress - x) / angleCoeff, Axis::Y);
                    cyls[0]->RotateInSystem(system, float(yAtPress - y) / angleCoeff, Axis::X);
                }
            }
            else{
                if (buttonState[GLFW_MOUSE_BUTTON_RIGHT] != GLFW_RELEASE)
                    pickedModel->TranslateInSystem(system, {-float(xAtPress - x) / moveCoeff, float(yAtPress - y) / moveCoeff, 0});
                if (buttonState[GLFW_MOUSE_BUTTON_MIDDLE] != GLFW_RELEASE)
                    pickedModel->RotateInSystem(system, float(xAtPress - x) / angleCoeff, Axis::Z);
                if (buttonState[GLFW_MOUSE_BUTTON_LEFT] != GLFW_RELEASE) {
                    pickedModel->RotateInSystem(system, float(xAtPress - x) / angleCoeff, Axis::Y);
                    pickedModel->RotateInSystem(system, float(yAtPress - y) / angleCoeff, Axis::X);
            }
            }
        } else {
           // camera->SetTout(cameraToutAtPress);
            if (buttonState[GLFW_MOUSE_BUTTON_RIGHT] != GLFW_RELEASE)
                root->TranslateInSystem(system, {-float(xAtPress - x) / moveCoeff/10.0f, float( yAtPress - y) / moveCoeff/10.0f, 0});
            if (buttonState[GLFW_MOUSE_BUTTON_MIDDLE] != GLFW_RELEASE)
                root->RotateInSystem(system, float(x - xAtPress) / 180.0f, Axis::Z);
            if (buttonState[GLFW_MOUSE_BUTTON_LEFT] != GLFW_RELEASE) {
                root->RotateInSystem(system, float(x - xAtPress) / angleCoeff, Axis::Y);
                root->RotateInSystem(system, float(y - yAtPress) / angleCoeff, Axis::X);
            }
        }
        xAtPress =  x;
        yAtPress =  y;
    }
}

void BasicScene::KeyCallback(Viewport* viewport, int x, int y, int key, int scancode, int action, int mods)
{
    auto system = camera->GetRotation().transpose();

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        switch (key) // NOLINT(hicpp-multiway-paths-covered)
        {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
            case GLFW_KEY_UP:
                cyls[pickedIndex]->RotateInSystem(system, 0.1f, Axis::X);
                break;
            case GLFW_KEY_DOWN:
                cyls[pickedIndex]->RotateInSystem(system, -0.1f, Axis::X);
                break;
            case GLFW_KEY_LEFT:
                cyls[pickedIndex]->RotateInSystem(system, 0.1f, Axis::Y);
                break;
            case GLFW_KEY_RIGHT:
                cyls[pickedIndex]->RotateInSystem(system, -0.1f, Axis::Y);
                break;
            case GLFW_KEY_W:
                camera->TranslateInSystem(system, {0, 0.1f, 0});
                break;
            case GLFW_KEY_S:
                camera->TranslateInSystem(system, {0, -0.1f, 0});
                break;
            case GLFW_KEY_A:
                camera->TranslateInSystem(system, {-0.1f, 0, 0});
                break;
            case GLFW_KEY_D:
                std::cout <<"Destination position:" <<std::endl<< sphere1->GetTranslation().transpose()<< std::endl;
                break;
            case GLFW_KEY_B:
                camera->TranslateInSystem(system, {0, 0, 0.1f});
                break;
            case GLFW_KEY_F:
                camera->TranslateInSystem(system, {0, 0, -0.1f});
                break;
            case GLFW_KEY_SPACE:
                if(!isFabrik)
                    isCCD = !isCCD;
                break;
            case GLFW_KEY_I:
                if(!isCCD)
                    isFabrik = !isFabrik;
                break;
            case GLFW_KEY_P:
                PrintRMats();
                break;
            case GLFW_KEY_N:
                if (pickedIndex == cyls.size() - 1) 
                    pickedIndex = 0;
                else 
                    pickedIndex++;
                pickedModel = cyls[pickedIndex];
                break;
            case GLFW_KEY_T:
                for (int i = 0; i < cyls.size(); i++) 
                    std::cout << "Tip" << i << " position: " << GetTipPos(i).transpose() << std::endl;
                break;

        }
    }
}

Eigen::Vector3d BasicScene::GetCenterDist(int i)
{
      Eigen::Vector3d l = Eigen::Vector3d(1.6f,0,0);
      Eigen::Vector3d res;
      res = cyls[i]->GetRotation().cast<double>()*l;
      res = res / 2;
      return res;  
      
}

Eigen::Vector3d BasicScene::GetTipPos(int i)
{
    //Eigen::Vector3d center =-cyls[i]->GetTin().translation().cast<double>();
    //Eigen::Vector4d center4=Eigen::Vector4d(center[0],center[1],center[2],1);
    //Eigen::Matrix4d trans = cyls[i]->GetTransform().cast<double>();
    //for (int j = i - 1; j >= 0; j--) {
    //    trans = cyls[j]->GetTransform().cast<double>() * trans;
    //}
    return cyls[i]->GetTranslation().cast<double>()+GetCenterDist(i);
}
Eigen::Vector3d BasicScene::GetStartPos(int i)
{
    //Eigen::Vector3d center = -cyls[i]->GetTin().translation().cast<double>();
    //Eigen::Vector4d center4 = Eigen::Vector4d(center[0], center[1], center[2], 1);
    //Eigen::Matrix4d trans = cyls[i]->GetTransform().cast<double>();
    //for (int j = i - 1; j >= 0; j--) {
    //    trans = cyls[j]->GetTransform().cast<double>() * trans;
    //}
    return cyls[i]->GetTranslation().cast<double>()-GetCenterDist(i);
}

Eigen::Matrix3d BasicScene::GetParentRot(int i) {
    Eigen::Matrix3d rMat = cyls[i]->GetRotation().cast<double>().inverse();
    for (int j = i - 1; j > 0; j--)
        rMat = rMat*cyls[j]->GetRotation().cast<double>().inverse();
    return rMat;
}

void BasicScene::CCD()
{
        Eigen::Vector3d dest = sphere1->GetTranslation().cast<double>();
        Eigen::Vector3d firstR = GetStartPos(0);
        double dist = (dest - firstR).norm();
        if (dist > 1.6 * cyls.size()) {
            std::cout << "cannot reach" << std::endl;
            isCCD = false;
            return;
        }
        int LinkID = cyls.size() - 1;
        while (LinkID >-1) {
            Eigen::Vector3d R = GetStartPos(LinkID);
            Eigen::Vector3d E = GetTipPos(cyls.size() - 1);
            Eigen::Vector3d RD = dest-R;
            Eigen::Vector3d RE = E - R;
            Eigen::Vector3d normal = RE.normalized().cross(RD.normalized());
            dist = (dest - E).norm();
            double dot = RD.normalized().dot(RE.normalized());
            if (dot > 1 || dot < -1) 
                dot = dot / abs(dot);
            double angle = acos(dot)/10;
            normal = cyls[LinkID]->GetRotation().cast<double>().inverse() * normal; 
            cyls[LinkID]->RotateByDegree(angle,normal.cast<float>());
            LinkID--;
        }
        if (dist < 0.05) {
            std::cout << "reached!  " << dist << std::endl;
            isCCD = false;
        }
}

void BasicScene::Fabrik() {
    std::vector<Eigen::Vector3d> P,oldP;
    std::vector<double> D;
    Eigen::Vector3d t = sphere1->GetTranslation().cast<double>();
    double tol = 0.05;
    double r, l;
    for (int i = 0; i < cyls.size(); i++)
        P.push_back(GetStartPos(i));
    P.push_back(GetTipPos(cyls.size() - 1));
    oldP = P;
    for (int i = 0; i < P.size() - 1; i++)
        D.push_back((P[i + 1] - P[i]).norm());
    double dist = (P[0]-t).norm();
    if (dist > 1.6 * cyls.size()) {
        std::cout << "cannot reach" << std::endl;
        isFabrik = false;
        return;
    }
    else {
        Eigen::Vector3d b = P[0];
        double difA = (P[P.size() - 1] - t).norm();
        if(difA > tol){
            P[P.size() - 1] = t;
            for (int i = P.size() - 2; i >= 0; i--) {
                r = (P[i+1] - P[i]).norm();
                l = D[i] / r;
                P[i] = (1 - l) * P[i+1] + l * P[i];
            }
            P[0] = b;
            for (int i = 0; i < P.size()-1; i++) {
                r = (P[i + 1] - P[i]).norm();
                l = D[i] / r;
                P[i+1] = (1 - l) * P[i] + l * P[i+1];
            }
            for (int i = 0; i < P.size()-1; i++) {
                Eigen::Vector3d Old = oldP[i+1]-oldP[i];
                Eigen::Vector3d New = P[i+1]-oldP[i];
                Eigen::Vector3d normal = New.normalized().cross(Old.normalized());
                double dot = New.normalized().dot(Old.normalized());
                if (dot > 1 || dot < -1)
                    dot = dot / abs(dot);
                double angle = acos(dot)/10;
                normal = GetParentRot(i) * normal;
                cyls[i]->RotateByDegree(-angle, normal.cast<float>());
            }
        }else{
            std::cout << "reached!  " << difA << std::endl;
            isFabrik = false;
            return;
        }
        
    }
}

void BasicScene::PrintRMats()
{   
    Eigen::Vector3d angles;
    if(pickedModel)
        angles= pickedModel->GetRotation().eulerAngles(2, 0, 2).cast<double>();
    else
        angles = root->GetRotation().eulerAngles(2, 0, 2).cast<double>();
    Eigen::MatrixXd rX(3, 3);
    Eigen::MatrixXd rY(3, 3);
    Eigen::MatrixXd rZ(3, 3);
    rX << cos(angles[0]), -sin(angles[0]), 0, sin(angles[0]), cos(angles[0]), 0, 0, 0, 1;
    rY << 1, 0, 0, 0, cos(angles[1]), -sin(angles[1]), 0, sin(angles[1]), cos(angles[1]);
    rZ << cos(angles[2]), -sin(angles[2]), 0, sin(angles[2]), cos(angles[2]), 0, 0, 0, 1;
    std::cout << "X Rotation:" <<std::endl<< rX << std::endl;
    std::cout << "Y Rotation:" << std::endl << rY << std::endl;
    std::cout << "Z Rotation:" << std::endl << rZ << std::endl;
}

int BasicScene::isPickedLink()
{
    for (int i = 0; i < cyls.size(); i++)
        if (pickedModel == cyls[i])
            return i;
    return -1;
}


