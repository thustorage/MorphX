// #include "llvm/Transforms/Utils/SMSched.h"
#include "SMSched.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <vector>
#include <map>

using namespace llvm;

PreservedAnalyses ForceInlinerPass::run(Module &M, ModuleAnalysisManager &AM) {
	if(M.getTargetTriple().getArch() != Triple::nvptx64) 
		return PreservedAnalyses::all();
	for (Function &F : M) {
		if(F.getCallingConv() != CallingConv::PTX_Kernel) {
			if (F.isDeclaration() || F.hasFnAttribute(Attribute::NoInline))
          		continue;
			F.addFnAttr(Attribute::AlwaysInline);
		}
	}
	return PreservedAnalyses::all();
}

Function* getNVVMFunc(Module *M, LLVMContext &ctx, const std::string &Name, Type *RetTy) {
    FunctionType *FuncTy = FunctionType::get(RetTy, {}, false);
    FunctionCallee Func = M->getOrInsertFunction(Name, FuncTy);
    Function *F = cast<Function>(Func.getCallee());
    F->addFnAttr(Attribute::NoUnwind); 
    return F;
}

cl::opt<bool> EnableDumpIR("enable-dump-ir", cl::desc("Enable dumping of IR"), cl::init(false));
cl::opt<std::string> DumpIRDir("dump-ir-dir", cl::desc("Directory to dump IR"), cl::init("."));

void checkAndWriteIR(Module &M, std::string prefix) {
	if(!EnableDumpIR)
		return;
	std::string fileName = DumpIRDir + "/" + prefix + "/" + M.getName().str() + "." + M.getTargetTriple().getArchName().str() + ".ll";
	size_t pos = fileName.find_last_of("/");
	std::string dirName = fileName.substr(0, pos);
	std::error_code EC;
	EC = sys::fs::create_directories(dirName, true);
	if(EC) {
		errs() << "Error creating directory: " << EC.message() << "\n";
		return;
	}
	raw_fd_ostream OS(fileName, EC, sys::fs::OF_None);
	if(EC) {
		errs() << "Error opening file: " << EC.message() << "\n";
		return;
	}
	M.print(OS, nullptr);
}

struct DeviceFuncInfo {
	std::string funcName;
	int argNum;
	Function *funcPtr;
};

// void registerNameWithArgNum(Module &M) {
// 	std::vector<DeviceFuncInfo> kernels;
// 	std::map<std::string, std::string> nameMap;
// 	LLVMContext &ctx = M.getContext();
// 	IRBuilder<> Builder(ctx);
// 	if(M.getTargetTriple().getArch() == Triple::nvptx64) 
// 		return;
	
// 	std::vector<Function*> cudaRegFuncs;
// 	Function *tFunc;
// 	if(tFunc = M.getFunction("__cuda_register_globals")) {
// 		cudaRegFuncs.push_back(tFunc);
// 	} 
// 	if(tFunc = M.getFunction("__cuda_module_ctor")) {
// 		cudaRegFuncs.push_back(tFunc);
// 	}

// 	for(Function *F : cudaRegFuncs) {
// 		for(Instruction &I : instructions(*F)) {
// 			if(CallInst *CI = dyn_cast<CallInst>(&I)) {
// 				Function *callee = CI->getCalledFunction();
// 				if(!callee || callee->getName() != "__cudaRegisterFunction") 
// 					continue;
// 				if(CI->arg_size() < 3){
// 					llvm::report_fatal_error("Insufficient arguments to __cudaRegisterFunction");
// 					return;
// 				}
// 				Value *arg2 = CI->getArgOperand(1)->stripPointerCasts();
// 				GlobalValue *funcGV = dyn_cast<GlobalValue>(arg2);
// 				if (!funcGV) continue;

// 				Value *arg3 = CI->getArgOperand(2)->stripPointerCasts();
// 				GlobalVariable *strGV = dyn_cast<GlobalVariable>(arg3);
// 				if (!strGV || !strGV->hasInitializer()) continue;

// 				Constant *initializer = strGV->getInitializer();
// 				ConstantDataArray *cda = dyn_cast<ConstantDataArray>(initializer);
// 				std::string kernelName = cda->getAsCString().str();
// 				nameMap[funcGV->getName().str()] = kernelName;
// 			}
// 		}
// 	}

// 	for(Function &F : M) {
// 		if(!F.getName().contains("__device_stub__"))
// 			continue;
// 		std::string funcName = F.getName().str();
// 		int argNum = 0;
// 		for(Instruction &I : instructions(F)) {
// 			if(CallInst *CI = dyn_cast<CallInst>(&I)) {
// 				Function *callee = CI->getCalledFunction();
// 				if(!callee || callee->getName() != "cudaLaunchKernel") 
// 					continue;
// 				Value *args = CI->getArgOperand(5);
// 				for(User *U : args->users()) {
// 					if(StoreInst *SI = dyn_cast<StoreInst>(U)) {
// 						++argNum;
// 					} else if(GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(U)) {
// 						++argNum;
// 					}
// 				}
// 			}
// 		}
// 		kernels.push_back({funcName, argNum, &F});
// 	}

// 	// Function *cudaLaunchKernelFunc = M.getFunction("cudaLaunchKernel");
// 	// Type *deviceFuncType = cudaLaunchKernelFunc->getFunctionType()->getParamType(0);
// 	FunctionType *regFuncType = FunctionType::get(Type::getVoidTy(ctx), {PointerType::get(ctx, 0), Type::getInt32Ty(ctx)}, false);
// 	Function *regFunc = Function::Create(regFuncType, GlobalValue::ExternalWeakLinkage, "register_name_with_arg_num", &M);
// 	Function *ctor = Function::Create(FunctionType::get(Type::getVoidTy(ctx), false), GlobalValue::InternalLinkage, "register_name_with_arg_num.ctor", &M);
// 	BasicBlock *ctorEntry = BasicBlock::Create(ctx, "entry", ctor);
// 	appendToGlobalCtors(M, ctor, 65535);

// 	Builder.SetInsertPoint(ctorEntry);
// 	for(auto info : kernels) {
// 		Constant *str = ConstantDataArray::getString(ctx, nameMap[info.funcName], true);
// 		GlobalVariable *GV = new GlobalVariable(M, str->getType(), true, GlobalValue::PrivateLinkage, str);
// 		Value *namePtr = Builder.CreateBitCast(GV, PointerType::get(ctx, 0));
// 		// Constant *funcPtr = ConstantExpr::getBitCast(info.funcPtr, deviceFuncType);
// 		Builder.CreateCall(regFunc, {namePtr, ConstantInt::get(Type::getInt32Ty(ctx), info.argNum)});
// 	}
// 	Builder.CreateRetVoid();
// }

PreservedAnalyses SMSchedPass::run(Module &M, ModuleAnalysisManager &AM) {

	checkAndWriteIR(M, "raw");

	std::vector<Function*> Funcs;
	for (Function &F : M) {
		if(F.getCallingConv() == CallingConv::PTX_Kernel)
			Funcs.push_back(&F);
	}
	LLVMContext &ctx = M.getContext();

	StructType *Dim3Ty = StructType::getTypeByName(ctx, "struct.dim3");
	if(!Dim3Ty) {
		std::vector<Type*> Elements;
		Elements.push_back(Type::getInt32Ty(ctx));
		Elements.push_back(Type::getInt32Ty(ctx));
		Elements.push_back(Type::getInt32Ty(ctx));
		Dim3Ty = StructType::create(Elements, "struct.dim3");
	}

	StringRef getsmid = "mov.u32 $0, %smid;";
	StringRef constraints = "=r";
	IRBuilder<> Builder(M.getContext());

	// Add a dynamic shared memory object if not present
	GlobalVariable *dynSmem = nullptr;
	for(GlobalVariable &GV : M.globals()) {
		if(GV.getAddressSpace() == 3 && GV.getLinkage() == GlobalValue::ExternalLinkage) {
			dynSmem = &GV;
			break;
		}
	}
	if(dynSmem == nullptr) {
		ArrayType *dynSharedTyp = ArrayType::get(Type::getInt32Ty(ctx), 0);
		dynSmem = new GlobalVariable(M, dynSharedTyp, false, 
			GlobalValue::ExternalLinkage, nullptr, "dynSharedMemSMSched", 
			nullptr, GlobalVariable::NotThreadLocal, 3);
		dynSmem->setAlignment(MaybeAlign(4));
	}

	NamedMDNode *Annotations = M.getNamedMetadata("nvvm.annotations");
	std::vector<MDNode*> NewAnnotationEntries;

	Function *tidX = getNVVMFunc(&M, ctx, "llvm.nvvm.read.ptx.sreg.tid.x", Type::getInt32Ty(ctx));
	Function *tidY = getNVVMFunc(&M, ctx, "llvm.nvvm.read.ptx.sreg.tid.y", Type::getInt32Ty(ctx));
	Function *tidZ = getNVVMFunc(&M, ctx, "llvm.nvvm.read.ptx.sreg.tid.z", Type::getInt32Ty(ctx));
	Function *barrier0 = getNVVMFunc(&M, ctx, "llvm.nvvm.barrier0", Type::getVoidTy(ctx));	
	Function *ctaX = M.getFunction("llvm.nvvm.read.ptx.sreg.ctaid.x");
	Function *ctaY = M.getFunction("llvm.nvvm.read.ptx.sreg.ctaid.y");
	Function *ctaZ = M.getFunction("llvm.nvvm.read.ptx.sreg.ctaid.z");
	Function *nctaX = M.getFunction("llvm.nvvm.read.ptx.sreg.nctaid.x");
	Function *nctaY = M.getFunction("llvm.nvvm.read.ptx.sreg.nctaid.y");
	Function *nctaZ = M.getFunction("llvm.nvvm.read.ptx.sreg.nctaid.z");

	for (Function *F : Funcs) {
		ValueToValueMapTy VMap;
		std::vector<Type*> argTypes;
		for(auto &arg : F->args()) {
			argTypes.push_back(arg.getType());
		}
		argTypes.push_back(PointerType::get(ctx, 0));
		argTypes.push_back(PointerType::getUnqual(ctx));
		argTypes.push_back(PointerType::getUnqual(ctx));
		argTypes.push_back(PointerType::getUnqual(ctx));
		argTypes.push_back(PointerType::getUnqual(ctx));
		argTypes.push_back(PointerType::getUnqual(ctx));
		FunctionType *nFuncTy = FunctionType::get(F->getReturnType(), argTypes, F->isVarArg());
		Function *nF = Function::Create(nFuncTy, F->getLinkage(), F->getAddressSpace(), F->getName() + "_pk", &M);
		nF->copyAttributesFrom(F);
		auto nArgIter = nF->arg_begin();
		for(auto &arg : F->args()) {
			VMap[&arg] = &*nArgIter;
			++nArgIter;
		}
		SmallVector<ReturnInst *, 8> Returns;
		CloneFunctionInto(nF, F, VMap, CloneFunctionChangeType::LocalChangesOnly, Returns);

		AttributeList attrs = nF->getAttributes();
		unsigned numArgs = nF->arg_size() - 6;
		attrs = attrs.addParamAttribute(ctx, numArgs, Attribute::getWithByValType(ctx, Dim3Ty));
		attrs = attrs.addParamAttribute(ctx, numArgs, Attribute::getWithAlignment(ctx, Align(4)));
		attrs = attrs.addParamAttribute(ctx, numArgs, Attribute::getWithCaptureInfo(ctx, CaptureInfo::none()));
		attrs = attrs.addParamAttribute(ctx, numArgs, Attribute::NoUndef);
		attrs = attrs.addParamAttribute(ctx, numArgs, Attribute::ReadOnly);
		attrs = attrs.addParamAttribute(ctx, numArgs + 1, Attribute::NoUndef); // agents
		attrs = attrs.addParamAttribute(ctx, numArgs + 2, Attribute::NoUndef); // fetched
		// attrs = attrs.addParamAttribute(ctx, numArgs + 2, Attribute::getWithCaptureInfo(ctx, CaptureInfo::none()));
		attrs = attrs.addParamAttribute(ctx, numArgs + 3, Attribute::NoUndef); // finished
		// attrs = attrs.addParamAttribute(ctx, numArgs + 3, Attribute::getWithCaptureInfo(ctx, CaptureInfo::none()));
		attrs = attrs.addParamAttribute(ctx, numArgs + 4, Attribute::NoUndef); // minSM
		attrs = attrs.addParamAttribute(ctx, numArgs + 4, Attribute::ReadOnly);
		// attrs = attrs.addParamAttribute(ctx, numArgs + 4, Attribute::getWithCaptureInfo(ctx, CaptureInfo::none()));
		attrs = attrs.addParamAttribute(ctx, numArgs + 5, Attribute::NoUndef); // maxSM
		attrs = attrs.addParamAttribute(ctx, numArgs + 5, Attribute::ReadOnly);
		nF->setAttributes(attrs);
		nF->removeFromParent();

		auto argIter = nF->arg_end();
		Argument *argGrid, *argAgents, *argBlockId, *argFin, *argMinSM, *argMaxSM; 
		argMaxSM = &*--argIter;
		argMinSM = &*--argIter;
		argFin = &*--argIter;
		argBlockId = &*--argIter;
		argAgents = &*--argIter;
		argGrid = &*--argIter;

		BasicBlock *oEntry = &nF->getEntryBlock();
		BasicBlock *blockGetSM = BasicBlock::Create(nF->getContext(), "get_sm", nF, oEntry);
		BasicBlock *decAgents = BasicBlock::Create(nF->getContext(), "dec_agents", nF, oEntry);
		BasicBlock *blockCheckMinSM = BasicBlock::Create(nF->getContext(), "smid_check_min", nF, oEntry);
		BasicBlock *blockCheckMaxSM = BasicBlock::Create(nF->getContext(), "smid_check_max", nF, oEntry);
		BasicBlock *blockMarkExit = BasicBlock::Create(nF->getContext(), "mark_exit", nF, oEntry);
		BasicBlock *blockWaitAgents = BasicBlock::Create(nF->getContext(), "wait_agents", nF, oEntry);
		BasicBlock *blockAtomicAdd = BasicBlock::Create(nF->getContext(), "block_atomicadd", nF, oEntry);
		BasicBlock *blockCheckBlock = BasicBlock::Create(nF->getContext(), "block_checkblock", nF, oEntry);
		BasicBlock *blockFin = BasicBlock::Create(nF->getContext(), "block_fin", nF);
		BasicBlock *blockFinAdd = BasicBlock::Create(nF->getContext(), "block_fin_add", nF);
		BasicBlock *blockRet = BasicBlock::Create(nF->getContext(), "block_ret", nF);

		Builder.SetInsertPoint(blockGetSM);
		InlineAsm *IA = InlineAsm::get(FunctionType::get(Builder.getInt32Ty(), false), getsmid, constraints, false);
		CallInst *callAsmSmId = Builder.CreateCall(IA);
		callAsmSmId->addAttributeAtIndex(AttributeList::ReturnIndex, Attribute::NoUndef);
		callAsmSmId->setTailCallKind(CallInst::TCK_Tail);

		Value *ptrGridY = Builder.CreateInBoundsGEP(Type::getInt8Ty(ctx), argGrid, 
			ConstantInt::get(Type::getInt64Ty(ctx), 4));
		Value *ptrGridZ = Builder.CreateInBoundsGEP(Type::getInt8Ty(ctx), argGrid, 
			ConstantInt::get(Type::getInt64Ty(ctx), 8));
		cast<GetElementPtrInst>(ptrGridY)->setHasNoUnsignedWrap(true);
		LoadInst *loadGridX = Builder.CreateAlignedLoad(Type::getInt32Ty(ctx), argGrid, Align(4));
		LoadInst *loadGridY = Builder.CreateAlignedLoad(Type::getInt32Ty(ctx), ptrGridY, Align(4));
		LoadInst *loadGridZ = Builder.CreateAlignedLoad(Type::getInt32Ty(ctx), ptrGridZ, Align(4));
		Value *mulGridXY = Builder.CreateMul(loadGridY, loadGridX);
		Value *nBlocks = Builder.CreateMul(loadGridZ, mulGridXY);

		CallInst *callTidX = Builder.CreateCall(tidX);
		CallInst *callTidY = Builder.CreateCall(tidY);
		CallInst *callTidZ = Builder.CreateCall(tidZ);
		callTidX->addAttributeAtIndex(AttributeList::ReturnIndex, Attribute::NoUndef);
		callTidY->addAttributeAtIndex(AttributeList::ReturnIndex, Attribute::NoUndef);
		callTidZ->addAttributeAtIndex(AttributeList::ReturnIndex, Attribute::NoUndef);
		callTidX->setTailCallKind(CallInst::TCK_Tail);
		callTidY->setTailCallKind(CallInst::TCK_Tail);
		callTidZ->setTailCallKind(CallInst::TCK_Tail);
		callTidX->setMetadata(LLVMContext::MD_range, MDNode::get(ctx, {ConstantAsMetadata::get(Builder.getInt32(0)), 
			ConstantAsMetadata::get(Builder.getInt32(1024))}));
		callTidY->setMetadata(LLVMContext::MD_range, MDNode::get(ctx, {ConstantAsMetadata::get(Builder.getInt32(0)), 
			ConstantAsMetadata::get(Builder.getInt32(1024))}));
		callTidZ->setMetadata(LLVMContext::MD_range, MDNode::get(ctx, {ConstantAsMetadata::get(Builder.getInt32(0)), 
			ConstantAsMetadata::get(Builder.getInt32(1024))}));
		Value *orYZ = Builder.CreateOr(callTidY, callTidZ);
		Value *orXYZ = Builder.CreateOr(orYZ, callTidX);
		Value *cmpThreadZ = Builder.CreateICmpEQ(orXYZ, ConstantInt::get(Type::getInt32Ty(ctx), 0));
		Builder.CreateCondBr(cmpThreadZ, decAgents, blockCheckBlock);

		Builder.SetInsertPoint(decAgents);
		AtomicRMWInst *atomicDec = Builder.CreateAtomicRMW(AtomicRMWInst::Sub, argAgents, 
			ConstantInt::get(Type::getInt32Ty(ctx), 1), MaybeAlign(4), AtomicOrdering::SequentiallyConsistent);
		Builder.CreateBr(blockCheckMinSM);
		
		Builder.SetInsertPoint(blockCheckMinSM);
		LoadInst *loadMinSM = Builder.CreateAlignedLoad(Type::getInt32Ty(ctx), argMinSM, Align(4));
		loadMinSM->setVolatile(true);
		Value *cmpSmMin = Builder.CreateICmpULT(callAsmSmId, loadMinSM);
		Builder.CreateCondBr(cmpSmMin, blockMarkExit, blockCheckMaxSM);

		Builder.SetInsertPoint(blockCheckMaxSM);
		LoadInst *loadMaxSM = Builder.CreateAlignedLoad(Type::getInt32Ty(ctx), argMaxSM, Align(4));
		loadMaxSM->setVolatile(true);
		Value *cmpSmMax = Builder.CreateICmpULT(callAsmSmId, loadMaxSM);
		Builder.CreateCondBr(cmpSmMax, blockAtomicAdd, blockMarkExit);

		Builder.SetInsertPoint(blockMarkExit);
		Value *castStore0 = Builder.CreateAddrSpaceCast(dynSmem, PointerType::get(ctx, 0));
		Builder.CreateAlignedStore(nBlocks, castStore0, Align(4), false);
		Builder.CreateBr(blockWaitAgents);

		Builder.SetInsertPoint(blockWaitAgents);
		AtomicRMWInst *loadAgents = Builder.CreateAtomicRMW(AtomicRMWInst::Add, argAgents,
			Builder.getInt32(0), MaybeAlign(4), AtomicOrdering::Acquire);
		Value *cmpAgentsZero = Builder.CreateICmpEQ(loadAgents, Builder.getInt32(0));
		Builder.CreateCondBr(cmpAgentsZero, blockCheckBlock, blockWaitAgents);

		Builder.SetInsertPoint(blockAtomicAdd);
		AtomicRMWInst *atomicAdd = Builder.CreateAtomicRMW(AtomicRMWInst::Add, argBlockId, 
			ConstantInt::get(Type::getInt32Ty(ctx), 1), MaybeAlign(4), AtomicOrdering::SequentiallyConsistent);
		Value *castStore = Builder.CreateAddrSpaceCast(dynSmem, PointerType::get(ctx, 0));
		Builder.CreateAlignedStore(atomicAdd, castStore, Align(4), false);
		Builder.CreateBr(blockCheckBlock);

		Builder.SetInsertPoint(blockCheckBlock);
		CallInst *callBarrier0 = Builder.CreateCall(barrier0);
		callBarrier0->setTailCallKind(CallInst::TCK_Tail);
		Value *castLoad = Builder.CreateAddrSpaceCast(dynSmem, PointerType::get(ctx, 0));
		LoadInst *loadBlockId = Builder.CreateAlignedLoad(Type::getInt32Ty(ctx), castLoad, Align(4));
		Value *cmpBlockId = Builder.CreateICmpSLT(loadBlockId, nBlocks);
		Builder.CreateCondBr(cmpBlockId, oEntry, blockRet);

		Builder.SetInsertPoint(oEntry->getFirstInsertionPt());		
		Value *freezeBlockId = Builder.CreateFreeze(loadBlockId);
		Value *freezeGridX = Builder.CreateFreeze(loadGridX);
		Value *divX = Builder.CreateUDiv(freezeBlockId, freezeGridX);
		Value *divXmulX = Builder.CreateMul(divX, freezeGridX);
		Value *blockIdx_x = Builder.CreateSub(freezeBlockId, divXmulX);
		Value *blockIdx_y = Builder.CreateURem(divX, loadGridY);
		Value *mulXY = Builder.CreateMul(loadGridY, loadGridX);
		Value *blockIdx_z = Builder.CreateUDiv(loadBlockId, mulXY);

		// Replace all calls to @llvm.nvvm.read.ptx.sreg.ctaid.xyz() with the above values
		std::vector<CallInst*> CItoErase;
		for(Instruction &I : instructions(nF)) {
			if(CallInst *CI = dyn_cast<CallInst>(&I)) {
				if(ctaX && CI->getCalledFunction() == ctaX) {
					CI->replaceAllUsesWith(blockIdx_x);
					CItoErase.push_back(CI);
				} else if(ctaY && CI->getCalledFunction() == ctaY) {
					CI->replaceAllUsesWith(blockIdx_y);
					CItoErase.push_back(CI);
				} else if(ctaZ && CI->getCalledFunction() == ctaZ) {
					CI->replaceAllUsesWith(blockIdx_z);
					CItoErase.push_back(CI);
				} else if(nctaX && CI->getCalledFunction() == nctaX) {
					CI->replaceAllUsesWith(loadGridX);
					CItoErase.push_back(CI);
				} else if(nctaY && CI->getCalledFunction() == nctaY) {
					CI->replaceAllUsesWith(loadGridY);
					CItoErase.push_back(CI);
				} else if(nctaZ && CI->getCalledFunction() == nctaZ) {
					CI->replaceAllUsesWith(loadGridZ);
					CItoErase.push_back(CI);
				}
			}
		}
		for(CallInst *CI : CItoErase) {
			CI->eraseFromParent();
		}
		CItoErase.clear();

		// Replace all ret instructions with a branch to blockFin
		std::vector<ReturnInst*> RItoErase;
		for(Instruction &I : instructions(nF)) {
			if(ReturnInst *RI = dyn_cast<ReturnInst>(&I)) {
				Builder.SetInsertPoint(RI);
				Builder.CreateBr(blockFin);
				RItoErase.push_back(RI);
			}
		}
		for(auto RI : RItoErase) {
			RI->eraseFromParent();
		}
		RItoErase.clear();

		Builder.SetInsertPoint(blockFin);
		Builder.CreateCondBr(cmpThreadZ, blockFinAdd, blockCheckBlock);

		Builder.SetInsertPoint(blockFinAdd);
		Builder.CreateAtomicRMW(AtomicRMWInst::Add, argFin, 
			ConstantInt::get(Type::getInt32Ty(ctx), 1), MaybeAlign(4), AtomicOrdering::SequentiallyConsistent);
		Builder.CreateBr(blockCheckMinSM);

		Builder.SetInsertPoint(blockRet);
		Builder.CreateRetVoid();
		
		if (const Comdat* OrigComdat = F->getComdat()) {
			std::string NewComdatName = nF->getName().str();
			Comdat* NewComdat = M.getOrInsertComdat(NewComdatName);

			NewComdat->setSelectionKind(OrigComdat->getSelectionKind());

			nF->setComdat(NewComdat);
		}

		M.getFunctionList().insertAfter(F->getIterator(), nF);
		std::string name = F->getName().str();
		// F->replaceAllUsesWith(nF);
		// F->eraseFromParent();

		// F->setName(name + "_pk1");
		// nF->setName(name);
		// F->setName(name + "_pk");
		// auto tComdat = F->getComdat();
		// F->setComdat(nF->getComdat());
		// nF->setComdat(tComdat);

		// Add to nvvm.annotations metadata
		if(!Annotations) {
			continue;
		}
		for (MDNode *AnnotationEntry : Annotations->operands()) {
			if (!AnnotationEntry || AnnotationEntry->getNumOperands() == 0) {
				continue;
			}
			auto *FuncMD = dyn_cast<ValueAsMetadata>(AnnotationEntry->getOperand(0));
			if (!FuncMD) {
				continue;
			}
			Function *OriginalKernel = dyn_cast<Function>(FuncMD->getValue());
			if (!OriginalKernel) {
				continue;
			}
			if (OriginalKernel != F) {
				continue;
			}

			MDNode *NewEntry;
			std::vector<Metadata*> NewOperands;

			NewOperands.push_back(ValueAsMetadata::get(nF));
			for (unsigned i = 1; i < AnnotationEntry->getNumOperands(); ++i) {
				NewOperands.push_back(AnnotationEntry->getOperand(i));
			}
			NewOperands.push_back(MDString::get(M.getContext(), "kernel"));
			NewOperands.push_back(ConstantAsMetadata::get(
				ConstantInt::get(Type::getInt32Ty(M.getContext()), 1)));
			NewEntry = MDNode::get(M.getContext(), NewOperands);
			NewAnnotationEntries.push_back(NewEntry);

			NewOperands.clear();
			NewOperands.push_back(ValueAsMetadata::get(F));
			for (unsigned i = 1; i < AnnotationEntry->getNumOperands(); ++i) {
				NewOperands.push_back(AnnotationEntry->getOperand(i));
			}
			NewOperands.push_back(MDString::get(M.getContext(), "kernel"));
			NewOperands.push_back(ConstantAsMetadata::get(
				ConstantInt::get(Type::getInt32Ty(M.getContext()), 1)));
			NewEntry = MDNode::get(M.getContext(), NewOperands);
			NewAnnotationEntries.push_back(NewEntry);
			break;
		}
	}

	if(Annotations) {
		Annotations->clearOperands();
		for (MDNode *NewEntry : NewAnnotationEntries) {
			Annotations->addOperand(NewEntry);
		}
	}

	// registerNameWithArgNum(M);

	checkAndWriteIR(M, "tr");

	// errs() << M << "\n";
    return PreservedAnalyses::all();
}

llvm::PassPluginLibraryInfo getSMSchedPluginInfo() {
	return {LLVM_PLUGIN_API_VERSION, "SMSched", LLVM_VERSION_STRING,
		[](PassBuilder &PB) {
			PB.registerPipelineStartEPCallback(
				[](ModulePassManager &MPM, OptimizationLevel Level) {
					MPM.addPass(ForceInlinerPass());
				});
			PB.registerOptimizerLastEPCallback(
				[](ModulePassManager &MPM, OptimizationLevel Level, llvm::ThinOrFullLTOPhase Phase) {
					MPM.addPass(SMSchedPass());
				});
		}};
  }

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
	return getSMSchedPluginInfo();
}
