/**
 * Author: Rohan Ramasamy
 * Date: 17/06/2017
 */

#include <vof/io/VTKWriter.h>

// #include <vtkXMLStructuredGridWriter.h>
// #include <vtkXMLUnstructuredGridWriter.h>
// #include <vtkStructuredGrid.h>
// #include <vtkUnstructuredGrid.h>
// #include <vtkDataSet.h>
// #include <vtkPointData.h>
// #include <vtkTetra.h>
// #include <vtkCellData.h>
// #include <vtkCellArray.h>
// #include <vtkDoubleArray.h>
// #include <vtkSmartPointer.h>


namespace vof {
    void
    VTKWriter::
    writeStructuredGridToFile(
        const std::shared_ptr<EulerianGrid> &grid,
        const double& time,
        const int& timeStep
        )
    {
    //     // Get grid states and geometry
    //     auto cellResolution = grid->cellResolution();
    //     auto cellDimensions = grid->cellSize();
    //     const auto densities = grid->densities();
    //     const auto pressures = grid->pressures();
    //     const auto internalEnergies = grid->internalEnergies();
    //     const auto velocities = grid->velocities();
        
    //     // Set VTK writer classes
    //     vtkSmartPointer<vtkXMLStructuredGridWriter> vtkWriter = vtkXMLStructuredGridWriter::New();
    //     vtkSmartPointer<vtkStructuredGrid> vtkGrid = vtkStructuredGrid::New();
    //     vtkGrid->SetDimensions(cellResolution[0], cellResolution[1], cellResolution[2]);
        
    //     vtkSmartPointer<vtkPoints> points = vtkPoints::New();
    //     points->Allocate(cellResolution[0] * cellResolution[1] * cellResolution[2]);
        
    //     vtkSmartPointer<vtkDoubleArray> rhoArray = vtkDoubleArray::New();
    //     rhoArray->SetNumberOfComponents(1);
    //     rhoArray->SetName("Density");
    //     vtkSmartPointer<vtkDoubleArray> pArray = vtkDoubleArray::New();
    //     pArray->SetNumberOfComponents(1);
    //     pArray->SetName("Pressure");
    //     vtkSmartPointer<vtkDoubleArray> eArray = vtkDoubleArray::New();
    //     eArray->SetNumberOfComponents(1);
    //     eArray->SetName("Internal_Energy");
    //     vtkSmartPointer<vtkDoubleArray> velArray = vtkDoubleArray::New();
    //     velArray->SetNumberOfComponents(3);
    //     velArray->SetName("Velocity");
        
    //     // Add data to VTK Output
    //     double x, y, z;
    //     for (int k = 0; k < cellResolution[2]; ++k) {
    //         for (int j = 0; j < cellResolution[1]; ++j) {
    //             for (int i = 0; i < cellResolution[0]; ++i) {
    //                 x = (i + 0.5) * cellDimensions[0];
    //                 y = (j + 0.5) * cellDimensions[1];
    //                 z = (k + 0.5) * cellDimensions[2];
                    
    //                 points->InsertNextPoint(x, y, z);
    //                 rhoArray->InsertNextTuple(&densities(i, j, k));
    //                 pArray->InsertNextTuple(&pressures(i, j, k));
    //                 velArray->InsertNextTuple3(velocities(i, j, k)[0],
    //                                            velocities(i, j, k)[1],
    //                                            velocities(i, j, k)[2]);
    //                 eArray->InsertNextTuple(&internalEnergies(i, j, k));
    //             }
    //         }
    //     }
        
    //     // Add to data to grid
    //     vtkGrid->SetPoints(points);
    //     vtkGrid->GetPointData()->AddArray(rhoArray);
    //     vtkGrid->GetPointData()->AddArray(pArray);
    //     vtkGrid->GetPointData()->AddArray(velArray);
    //     vtkGrid->GetPointData()->AddArray(eArray);
        
    //     // Write to file
    //     std::stringstream ss;
    //     ss << "ts" << timeStep << ".vts";
    //     vtkWriter->SetInputData(vtkGrid);
    //     vtkWriter->SetFileName(ss.str().c_str());
    //     vtkWriter->SetDataModeToAscii();
    //     vtkWriter->Write();
    }
}
