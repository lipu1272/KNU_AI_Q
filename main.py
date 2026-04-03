from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import pandas as pd

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 선택사항: 있으면 더 자연스럽게 설명
USE_LLM = True
try:
    from langchain_groq import ChatGroq
except Exception:
    USE_LLM = False


# =========================================================
# 1. 데이터 구조
# =========================================================

@dataclass
class KBRow:
    id: int
    category: str
    sub_category: str
    title: str
    content: str

    def to_text(self) -> str:
        return (
            f"[ID {self.id}] {self.title}\n"
            f"category: {self.category}\n"
            f"sub_category: {self.sub_category}\n"
            f"content: {self.content}"
        )


@dataclass
class AgentResult:
    success: bool
    query_type: str
    decision: str
    summary: str
    retrieved_rules: List[KBRow] = field(default_factory=list)
    next_node: Optional[str] = None
    reasons: List[str] = field(default_factory=list)


# =========================================================
# 2. FAISS 기반 KB
# =========================================================

class BankingKnowledgeBase:
    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

        self.df = pd.read_csv(csv_path)

        required_cols = {"id", "category", "sub_category", "title", "content"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV 필수 컬럼 누락: {missing}")

        self.rows: List[KBRow] = [
            KBRow(
                id=int(row["id"]),
                category=str(row["category"]),
                sub_category=str(row["sub_category"]),
                title=str(row["title"]),
                content=str(row["content"]),
            )
            for _, row in self.df.iterrows()
        ]
        self.id_map: Dict[int, KBRow] = {row.id: row for row in self.rows}

        # 로컬 임베딩 모델
        # 처음 실행 시 모델 다운로드 때문에 조금 걸릴 수 있음
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.documents = self._build_documents()
        self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)

    def _build_documents(self) -> List[Document]:
        docs = []
        for row in self.rows:
            text = (
                f"ID: {row.id}\n"
                f"category: {row.category}\n"
                f"sub_category: {row.sub_category}\n"
                f"title: {row.title}\n"
                f"content: {row.content}"
            )

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "id": row.id,
                        "category": row.category,
                        "sub_category": row.sub_category,
                        "title": row.title,
                    },
                )
            )
        return docs

    def get_by_id(self, row_id: int) -> Optional[KBRow]:
        return self.id_map.get(row_id)

    def get_rows(self, ids: List[int]) -> List[KBRow]:
        return [self.id_map[i] for i in ids if i in self.id_map]

    def search(self, query: str, top_k: int = 5) -> List[KBRow]:
        docs = self.vectorstore.similarity_search(query, k=top_k)

        result_rows = []
        seen_ids = set()

        for doc in docs:
            row_id = doc.metadata.get("id")
            if row_id in self.id_map and row_id not in seen_ids:
                result_rows.append(self.id_map[row_id])
                seen_ids.add(row_id)

        return result_rows

    def search_with_score(self, query: str, top_k: int = 5):
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)

        formatted = []
        seen_ids = set()

        for doc, score in results:
            row_id = doc.metadata.get("id")
            if row_id in self.id_map and row_id not in seen_ids:
                formatted.append((self.id_map[row_id], score))
                seen_ids.add(row_id)

        return formatted


# =========================================================
# 3. 선택적 LLM 설명기
# =========================================================

class LLMReporter:
    def __init__(self):
        self.enabled = False
        self.llm = None

        groq_api_key = os.getenv("GROQ_API_KEY")
        if USE_LLM and groq_api_key:
            try:
                self.llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0,
                    groq_api_key=groq_api_key,
                )
                self.enabled = True
            except Exception:
                self.enabled = False

    def summarize(self, user_request: str, rules: List[KBRow], extra_instruction: str = "") -> str:
        if not rules:
            return "관련 규정을 찾지 못했습니다."

        if not self.enabled or self.llm is None:
            lines = []
            if extra_instruction:
                lines.append(extra_instruction)
            lines.append("")
            lines.append("근거 규정:")
            for r in rules:
                lines.append(f"- ID {r.id} | {r.title}: {r.content}")
            return "\n".join(lines)

        context = "\n\n".join([r.to_text() for r in rules])

        prompt = f"""
당신은 금융 규정 기반 에이전트 설명기입니다.
아래 문맥만 사용해서 한국어로 답하세요.

[사용자 요청]
{user_request}

[규정 문맥]
{context}

[추가 지시]
{extra_instruction}

반드시 포함할 것:
1. 판단 결과
2. 근거 규정 ID
3. 다음 노드 또는 추가 조치
""".strip()

        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)


# =========================================================
# 4. 규칙 기반 에이전트
# =========================================================

class BankingAgent:
    def __init__(self, kb: BankingKnowledgeBase):
        self.kb = kb
        self.reporter = LLMReporter()

    def check_transfer(self, user_grade: str, amount_krw: int) -> AgentResult:
        rules = self.kb.get_rows([3, 4, 5])

        user_grade = user_grade.strip().upper()
        success = True
        decision = "이체 가능"
        next_node = "Execution"
        reasons = []

        if user_grade == "일반":
            if amount_krw > 5_000_000:
                success = False
                decision = "이체 불가"
                next_node = "Compliance_Review"
                reasons.append("일반 등급의 1일 최대 이체 한도(500만 원)를 초과했습니다. [ID 3]")
            elif amount_krw > 2_000_000:
                success = False
                decision = "이체 불가"
                next_node = "Compliance_Review"
                reasons.append("일반 등급의 1회 이체 한도(200만 원)를 초과했습니다. [ID 3]")
            else:
                reasons.append("일반 등급의 한도 내 금액입니다. [ID 3]")

        elif user_grade == "VIP":
            if amount_krw > 50_000_000:
                success = False
                decision = "이체 불가"
                next_node = "Compliance_Review"
                reasons.append("VIP 등급의 1일 최대 이체 한도(5,000만 원)를 초과했습니다. [ID 4]")
            elif amount_krw >= 10_000_000:
                next_node = "Security_Check"
                reasons.append("VIP 고액 이체 구간으로 Security_Check가 필요합니다. [ID 4]")
                reasons.append("생체 인증과 SMS 인증을 수행해야 합니다. [ID 5]")
            else:
                reasons.append("VIP 등급 한도 내 금액입니다. [ID 4]")
        else:
            success = False
            decision = "이체 불가"
            next_node = None
            reasons.append("사용자 등급은 'VIP' 또는 '일반'만 입력 가능합니다.")

        summary = self.reporter.summarize(
            user_request=f"user_grade={user_grade}, amount_krw={amount_krw}",
            rules=rules,
            extra_instruction="이체 가능 여부와 MFA 필요 여부를 설명하세요."
        )

        return AgentResult(
            success=success,
            query_type="transfer_limit_check",
            decision=decision,
            summary=summary,
            retrieved_rules=rules,
            next_node=next_node,
            reasons=reasons,
        )

    def analyze_blocked_transaction(
        self,
        repeated_small_payments_in_1h: int,
        unusual_foreign_ip: bool,
    ) -> AgentResult:
        rules = self.kb.get_rows([9, 10])
        reasons = []
        blocked = False

        if repeated_small_payments_in_1h >= 5:
            blocked = True
            reasons.append("최근 1시간 내 5회 이상의 소액 반복 결제 조건에 해당합니다. [ID 9]")

        if unusual_foreign_ip:
            blocked = True
            reasons.append("평소 접속하지 않던 해외 IP 접근 조건에 해당합니다. [ID 9]")

        if blocked:
            success = False
            decision = "거래 차단"
            next_node = "고객센터 연결"
            reasons.append("차단 후 실행 노드를 건너뛰고 고객센터 연결 노드로 이동합니다. [ID 10]")
        else:
            success = True
            decision = "차단 사유 없음"
            next_node = "Validation/Execution"
            reasons.append("ID 9의 차단 조건을 만족하지 않습니다.")

        summary = self.reporter.summarize(
            user_request=(
                f"repeated_small_payments_in_1h={repeated_small_payments_in_1h}, "
                f"unusual_foreign_ip={unusual_foreign_ip}"
            ),
            rules=rules,
            extra_instruction="차단 근거와 차단 이후 이동 노드를 설명하세요."
        )

        return AgentResult(
            success=success,
            query_type="fds_history_analysis",
            decision=decision,
            summary=summary,
            retrieved_rules=rules,
            next_node=next_node,
            reasons=reasons,
        )

    def compliance_gate(
        self,
        request_type: str,
        annual_overseas_remit_usd: float = 0.0,
        annual_income_krw: int = 0,
        annual_debt_service_krw: int = 0,
        investment_profile: str = "",
        product_risk: str = "",
    ) -> AgentResult:
        rules = self.kb.get_rows([26, 30, 39])
        reasons = []
        success = True
        decision = "승인 가능"
        next_node = "Approved_Path"

        if request_type == "해외송금":
            if annual_overseas_remit_usd > 50_000:
                success = False
                decision = "거절"
                next_node = "Document_Upload"
                reasons.append("증빙 없는 연간 해외송금 한도(USD 50,000)를 초과했습니다. [ID 26]")
                reasons.append("법적 근거: 외국환거래 관련 한도 규정입니다. [ID 26]")

                return AgentResult(
                    success=success,
                    query_type="compliance_gate",
                    decision=decision,
                    summary=self.reporter.summarize(
                        user_request=f"해외송금 요청, annual_overseas_remit_usd={annual_overseas_remit_usd}",
                        rules=rules,
                        extra_instruction="ID 26 단계에서 거절된 이유를 설명하세요."
                    ),
                    retrieved_rules=rules,
                    next_node=next_node,
                    reasons=reasons,
                )

            reasons.append("연간 해외송금 한도 내입니다. [ID 26]")

        if request_type == "대출":
            if annual_income_krw <= 0:
                success = False
                decision = "거절"
                next_node = "Credit_Score"
                reasons.append("연소득이 0 이하라 DSR 산정이 불가능합니다. [ID 30]")

                return AgentResult(
                    success=success,
                    query_type="compliance_gate",
                    decision=decision,
                    summary=self.reporter.summarize(
                        user_request=(
                            f"대출 요청, annual_income_krw={annual_income_krw}, "
                            f"annual_debt_service_krw={annual_debt_service_krw}"
                        ),
                        rules=rules,
                        extra_instruction="ID 30 단계에서 거절된 이유를 설명하세요."
                    ),
                    retrieved_rules=rules,
                    next_node=next_node,
                    reasons=reasons,
                )

            dsr = annual_debt_service_krw / annual_income_krw
            if dsr > 0.40:
                success = False
                decision = "거절"
                next_node = "Credit_Score"
                reasons.append(f"DSR이 40%를 초과했습니다. 계산값: {dsr:.2%} [ID 30]")
                reasons.append("법적/내부 심사 근거: 대출 검증 노드의 DSR 기준입니다. [ID 30]")

                return AgentResult(
                    success=success,
                    query_type="compliance_gate",
                    decision=decision,
                    summary=self.reporter.summarize(
                        user_request=(
                            f"대출 요청, annual_income_krw={annual_income_krw}, "
                            f"annual_debt_service_krw={annual_debt_service_krw}, dsr={dsr:.2%}"
                        ),
                        rules=rules,
                        extra_instruction="ID 30 단계에서 거절된 이유를 설명하세요."
                    ),
                    retrieved_rules=rules,
                    next_node=next_node,
                    reasons=reasons,
                )

            reasons.append(f"DSR 기준 충족: {dsr:.2%} <= 40% [ID 30]")

        if request_type == "투자":
            if investment_profile == "안정형" and product_risk == "고위험":
                success = False
                decision = "거절"
                next_node = "상품추천 차단"
                reasons.append("안정형 투자자에게 고위험 상품 추천은 불가합니다. [ID 39]")
                reasons.append("법적 근거: 투자 적합성 원칙 위반입니다. [ID 39]")

                return AgentResult(
                    success=success,
                    query_type="compliance_gate",
                    decision=decision,
                    summary=self.reporter.summarize(
                        user_request=(
                            f"투자 요청, investment_profile={investment_profile}, "
                            f"product_risk={product_risk}"
                        ),
                        rules=rules,
                        extra_instruction="ID 39 단계에서 거절된 이유를 설명하세요."
                    ),
                    retrieved_rules=rules,
                    next_node=next_node,
                    reasons=reasons,
                )

            reasons.append("투자 성향과 상품 위험도 간 충돌이 없습니다. [ID 39]")

        summary = self.reporter.summarize(
            user_request=(
                f"request_type={request_type}, annual_overseas_remit_usd={annual_overseas_remit_usd}, "
                f"annual_income_krw={annual_income_krw}, annual_debt_service_krw={annual_debt_service_krw}, "
                f"investment_profile={investment_profile}, product_risk={product_risk}"
            ),
            rules=rules,
            extra_instruction="순차 검증 결과와 승인 가능 여부를 설명하세요."
        )

        return AgentResult(
            success=success,
            query_type="compliance_gate",
            decision=decision,
            summary=summary,
            retrieved_rules=rules,
            next_node=next_node,
            reasons=reasons,
        )


# =========================================================
# 5. 출력 함수
# =========================================================

def print_result(result: AgentResult) -> None:
    print("\n" + "=" * 70)
    print(f"[질의 유형] {result.query_type}")
    print(f"[판단] {result.decision}")
    print(f"[다음 노드] {result.next_node}")

    print("\n[설명]")
    print(result.summary)

    print("\n[세부 근거]")
    for reason in result.reasons:
        print(f"- {reason}")

    print("\n[참조 규정]")
    for row in result.retrieved_rules:
        print(f"- ID {row.id} | {row.title}")

    print("=" * 70 + "\n")


# =========================================================
# 6. 실행부
# =========================================================

def main():
    csv_path = "dataset.csv"
    kb = BankingKnowledgeBase(csv_path)
    agent = BankingAgent(kb)

    while True:
        print("===== Neo-Finance FAISS RAG Agent =====")
        print("1. 이체 가능 여부 + MFA 확인")
        print("2. ID 9 차단 사유 + ID 10 다음 노드 설명")
        print("3. 해외송금/대출/투자 순차 검증")
        print("4. FAISS 자유 검색 테스트")
        print("0. 종료")

        choice = input("메뉴 선택: ").strip()

        if choice == "1":
            user_grade = input("사용자 등급 입력 (VIP/일반): ").strip()
            amount_krw = int(input("요청 금액 입력(원): ").strip())
            result = agent.check_transfer(user_grade=user_grade, amount_krw=amount_krw)
            print_result(result)

        elif choice == "2":
            repeated_small = int(input("최근 1시간 소액 반복 결제 횟수: ").strip())
            unusual_ip = input("평소와 다른 해외 IP 접근 여부 (y/n): ").strip().lower() == "y"
            result = agent.analyze_blocked_transaction(
                repeated_small_payments_in_1h=repeated_small,
                unusual_foreign_ip=unusual_ip,
            )
            print_result(result)

        elif choice == "3":
            req_type = input("요청 종류 입력 (해외송금/대출/투자): ").strip()

            if req_type == "해외송금":
                amount_usd = float(input("연간 누적 해외송금액(USD): ").strip())
                result = agent.compliance_gate(
                    request_type=req_type,
                    annual_overseas_remit_usd=amount_usd,
                )
                print_result(result)

            elif req_type == "대출":
                income = int(input("연소득(원): ").strip())
                debt_service = int(input("연간 원리금 상환액 합계(원): ").strip())
                result = agent.compliance_gate(
                    request_type=req_type,
                    annual_income_krw=income,
                    annual_debt_service_krw=debt_service,
                )
                print_result(result)

            elif req_type == "투자":
                profile = input("투자 성향 입력 (예: 안정형): ").strip()
                risk = input("상품 위험도 입력 (예: 고위험): ").strip()
                result = agent.compliance_gate(
                    request_type=req_type,
                    investment_profile=profile,
                    product_risk=risk,
                )
                print_result(result)
            else:
                print("지원하지 않는 요청 종류입니다.")

        elif choice == "4":
            query = input("검색어 입력: ").strip()
            results = kb.search_with_score(query, top_k=5)

            if not results:
                print("\n검색 결과가 없습니다.\n")
            else:
                print("\n[FAISS 검색 결과]")
                for row, score in results:
                    print(f"\n- score={score:.4f} | ID {row.id} | {row.title}")
                    print(f"  category={row.category}, sub_category={row.sub_category}")
                    print(f"  content={row.content}")

        elif choice == "0":
            print("프로그램 종료")
            break

        else:
            print("올바른 메뉴를 선택하세요.")


if __name__ == "__main__":
    main()